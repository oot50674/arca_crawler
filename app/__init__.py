import os
import secrets
from datetime import datetime
from pathlib import Path

from flask import Flask


def create_app():
    """Flask 앱 팩토리를 간소화하여 크롤러 서비스만 담습니다."""
    app = Flask(__name__)

    def _ensure_secret_key():
        """
        `.env`에 SECRET_KEY가 없으면 생성하여 저장합니다.
        프로젝트 루트(앱 폴더 상위)에 `.env` 파일을 둡니다.
        """
        project_root = os.path.join(app.root_path, '..')
        env_path = os.path.join(project_root, '.env')

        secret = None
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.startswith('SECRET_KEY='):
                            secret = line.split('=', 1)[1]
                            break
            except Exception:
                secret = None

        if not secret:
            secret = secrets.token_urlsafe(32)
            try:
                os.makedirs(os.path.dirname(env_path), exist_ok=True)
                with open(env_path, 'a', encoding='utf-8') as f:
                    f.write('\nSECRET_KEY=' + secret + '\n')
            except Exception:
                pass

        return secret

    app.config['SECRET_KEY'] = _ensure_secret_key()
    app.config['BACKUP_ROOT'] = Path(app.root_path).parent / "backups"
    app.config['BACKUP_ROOT'].mkdir(parents=True, exist_ok=True)

    @app.template_filter('datetimeformat')
    def datetimeformat(value, fmt="%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.fromtimestamp(float(value)).strftime(fmt)
        except Exception:
            return ""

    # 블루프린트(라우트) 등록
    from .routes import main
    app.register_blueprint(main)

    return app
