"""Save a human-provided token on the server; network validation is separate."""
import getpass
import os
from pathlib import Path
import tempfile


def save_token(home, token):
    if not token or any(c.isspace() for c in token):
        raise ValueError('Token must be nonempty and contain no whitespace')
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    home.chmod(0o700)
    with tempfile.NamedTemporaryFile(mode='w', dir=home, delete=False) as f:
        temporary = Path(f.name)
        f.write(token + '\n')
    temporary.chmod(0o600)
    os.replace(temporary, home / 'token')


if __name__ == '__main__':
    token = getpass.getpass('请输入你本人已获Llama授权账号的HF token（不回显）：').strip()
    home = Path(__file__).parent / 'hf_auth'
    save_token(home, token)
    del token
    print('已保存到本项目独立认证目录，尚未进行官方访问验证。')
    print('请回到对话回复：项目认证已更新。')
