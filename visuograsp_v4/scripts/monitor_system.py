"""System monitor for long V4 training runs on the Windows laptop (logging + safe pause/stop).

Every --interval seconds: ACPI thermal zone temperature, % passive limit (thermal throttling of
the CPU; 100 = none), throttle reasons, % processor performance / utility, free memory, and per
run: process alive, last logged step, training steps/s, gate state. Written to
outputs/rl/<group>/system_monitor.csv; events are printed (one line each).

Safety rules (serious thermal throttling or low memory):
  temperature >= --temp-pause-k for --sustain samples, or % passive limit <= --passive-pause for
  --sustain samples, or free memory < 1 GB  -> pause_request.json in every active run (the run
  saves a checkpoint and sleeps; replay buffer stays in memory);
  recovery (temp <= --temp-resume-k and passive limit 100 for 3 samples) -> request removed;
  paused longer than --max-pause-min -> stop_request.json (run saves and stops).
Counters need the interactive user context (start from PowerShell, not from a sandboxed shell).

    python -m visuograsp_v4.scripts.monitor_system --runs wave1/sac_scratch_s11 wave1/sac_demo_s11
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from visuograsp_v4.config import OUTPUT_ROOT

PS = r"""
$paths = @('\Thermal Zone Information(*)\Temperature', '\Thermal Zone Information(*)\% Passive Limit',
           '\Thermal Zone Information(*)\Throttle Reasons', '\Processor Information(_Total)\% Processor Performance',
           '\Processor Information(_Total)\% Processor Utility')
$r = Get-Counter -Counter $paths -ErrorAction SilentlyContinue
$o = @{}
foreach ($x in $r.CounterSamples) { $o[$x.Path] = [string]$x.CookedValue }
$o | ConvertTo-Json -Compress
"""


COUNTER_COLS = ('temp_k', 'temp_c', 'passive_limit_pct', 'throttle_reasons', 'cpu_perf_pct', 'cpu_util_pct')
RUN_COLS = ('step', 'steps_per_s', 'q_max', 'alive', 'gate_waiting', 'paused', 'finished')


def counters() -> dict:
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', PS],
                             capture_output=True, text=True, timeout=60)
        raw = json.loads(out.stdout or '{}')
        if not raw:
            return {'counter_error': (out.stderr or 'empty counter result').strip()[:100]}
    except Exception as exc:                      # never let the monitor die on a counter error
        return {'counter_error': str(exc)[:100]}
    res = {}
    for path, val in raw.items():
        v = float(str(val).replace(',', '.'))
        p = path.lower()
        if 'thermal zone' in p and p.endswith('\\temperature'):
            res['temp_k'] = max(res.get('temp_k', 0.0), v)
        elif 'passive limit' in p:
            res['passive_limit_pct'] = min(res.get('passive_limit_pct', 100.0), v)
        elif 'throttle reasons' in p:
            res['throttle_reasons'] = max(res.get('throttle_reasons', 0.0), v)
        elif 'processor performance' in p:
            res['cpu_perf_pct'] = v
        elif 'processor utility' in p:
            res['cpu_util_pct'] = v
    if 'temp_k' in res:
        res['temp_c'] = res['temp_k'] - 273.15
    return res


def run_state(run: Path) -> dict:
    st = {'finished': (run / 'finished.json').exists(), 'gate_waiting': (run / 'gate_waiting.json').exists(),
          'paused': (run / 'pause_request.json').exists()}
    cfg = run / 'run_config.json'
    if cfg.exists():
        pid = json.loads(cfg.read_text(encoding='utf-8')).get('pid')
        st['pid'] = pid
        st['alive'] = bool(pid and psutil.pid_exists(pid))
    q = run / 'qhealth.csv'
    if q.exists():
        try:
            last = pd.read_csv(q).iloc[-1]
            st.update(step=int(last.step), steps_per_s=float(last.steps_per_s), q_max=float(last.q_max),
                      logged_at=q.stat().st_mtime)
        except Exception:
            pass
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--group', default='wave1')
    ap.add_argument('--interval', type=float, default=60.0)
    ap.add_argument('--temp-pause-k', type=float, default=368.15)     # 95 C
    ap.add_argument('--temp-resume-k', type=float, default=358.15)    # 85 C
    ap.add_argument('--passive-pause', type=float, default=70.0)
    ap.add_argument('--sustain', type=int, default=5)
    ap.add_argument('--max-pause-min', type=float, default=30.0)
    ap.add_argument('--stall-min', type=float, default=20.0)
    args = ap.parse_args()
    runs = [OUTPUT_ROOT / 'rl' / r for r in args.runs]
    log = OUTPUT_ROOT / 'rl' / args.group / 'system_monitor.csv'
    hot = good = 0
    paused_since = None
    stall_warned = set()
    print(f'[MON] start {datetime.now():%H:%M:%S}, runs {args.runs}', flush=True)
    while True:
        c = counters()
        mem = psutil.virtual_memory()
        row = {k: c.get(k) for k in COUNTER_COLS}
        row.update(time=datetime.now().isoformat(timespec='seconds'), counter_error=c.get('counter_error'),
                   mem_available_gb=mem.available / 2 ** 30, cpu_percent_psutil=psutil.cpu_percent(interval=1.0))
        states = {r.name: run_state(r) for r in runs}
        for name, st in states.items():
            for k in RUN_COLS:
                row[f'{name}_{k}'] = st.get(k)
        serious = ((c.get('temp_k', 0) >= args.temp_pause_k) or (c.get('passive_limit_pct', 100) <= args.passive_pause)
                   or mem.available < 2 ** 30)
        hot = hot + 1 if serious else 0
        recovered = c.get('temp_k', 999) <= args.temp_resume_k and c.get('passive_limit_pct', 0) >= 100 and mem.available >= 2 ** 30
        good = good + 1 if recovered else 0
        active = [r for r in runs if not states[r.name]['finished']]
        if hot >= args.sustain and paused_since is None and active:
            paused_since = time.time()
            for r in active:
                (r / 'pause_request.json').write_text(json.dumps({'by': 'monitor_system', 'row': row}, default=str),
                                                      encoding='utf-8')
            print(f'[MON] PAUSE requested (serious throttling/memory): {c} mem {mem.available / 2 ** 30:.1f} GB', flush=True)
        elif paused_since is not None:
            if good >= 3:
                for r in runs:
                    (r / 'pause_request.json').unlink(missing_ok=True)
                print(f'[MON] RESUME after {(time.time() - paused_since) / 60:.1f} min: {c}', flush=True)
                paused_since = None
            elif time.time() - paused_since > 60 * args.max_pause_min:
                for r in active:
                    (r / 'stop_request.json').write_text(json.dumps({'by': 'monitor_system', 'reason':
                                                                     'thermal pause exceeded limit', 'row': row},
                                                                    default=str), encoding='utf-8')
                print('[MON] STOP requested: pause exceeded limit', flush=True)
                paused_since = None
        row['pause_active'] = paused_since is not None
        cols = (['time'] + list(COUNTER_COLS) + ['counter_error', 'mem_available_gb', 'cpu_percent_psutil']
                + [f'{r.name}_{k}' for r in runs for k in RUN_COLS] + ['pause_active'])
        pd.DataFrame([row])[cols].to_csv(log, mode='a', header=not log.exists(), index=False)   # fixed column order
        for name, st in states.items():
            if st.get('alive') is False and not st['finished'] and name not in stall_warned:
                stall_warned.add(name)
                print(f'[MON] PROCESS DEAD without finished.json: {name}', flush=True)
            if (st.get('alive') and not st['gate_waiting'] and not st['paused'] and 'logged_at' in st
                    and time.time() - st['logged_at'] > 60 * args.stall_min and name not in stall_warned):
                stall_warned.add(name)
                print(f'[MON] STALL: {name} no qhealth row for {args.stall_min} min', flush=True)
        if all(states[r.name]['finished'] for r in runs):
            print('[MON] all runs finished; exit', flush=True)
            return
        time.sleep(max(1.0, args.interval - 3))


if __name__ == '__main__':
    main()
