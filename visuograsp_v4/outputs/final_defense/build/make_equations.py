"""Equation images for the slides (matplotlib mathtext, transparent PNG). The equations follow the code:
env.py (reward v4r1), controller.py (wrist / finger update), wave5.py (E6 actor objective)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'presentation' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
EQ = {
    'eq_reward': r'$r_t=\gamma\,\Phi(s_{t+1})-\Phi(s_t)\;+\;0.1\cdot\mathbb{I}_{\rm hold}\;+\;10\cdot\mathbb{I}_{\rm success}'
                 r'\;-\;3\cdot\mathbb{I}_{\rm drop}\;-\;c_{\rm table}-c_{\rm effort}-5\cdot\mathbb{I}_{\rm sim\,fail}$',
    'eq_phi': r'$\Phi=\Phi_{\rm reach}+\Phi_{\rm align}+\Phi_{\rm contact}+3\,{\rm clip}(h/0.10\,{\rm m})+\Phi_{\rm push},\quad \gamma=0.99$',
    'eq_wrist': r'$p\leftarrow p+0.01\,{\rm m}\cdot a_{16:19}\qquad q\leftarrow \frac{q\otimes\exp(0.06\,a_{19:22})}'
                r'{\Vert q\otimes\exp(0.06\,a_{19:22})\Vert}\qquad \theta^{*}\leftarrow\theta^{*}+0.055\,a_{0:16}$',
    'eq_e6': r'$\mathcal{L}_\pi=\mathbb{E}_{B}\!\left[w\,\Vert\tanh\mu(s)-a\Vert^2\right]+0.5\,\mathbb{E}_{D}\!\left[\Vert\tanh\mu(s_d)-a^{*}_d\Vert^2\right],'
             r'\quad w\propto\exp\!\left(\frac{A(s,a)}{\tau\,\sigma_A}\right)$',
}


def render(name, tex, fs=22, color='#1f2a37'):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=fs, color=color)
    fig.savefig(OUT / f'{name}.png', dpi=250, transparent=True, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams['mathtext.fontset'] = 'cm'
    for k, v in EQ.items():
        render(k, v)
    print('equations written to', OUT)
