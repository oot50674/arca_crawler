# Arca Crawler
<img width="2489" height="1626" alt="image" src="https://github.com/user-attachments/assets/92c77681-430a-49d8-a3e6-755a27f373dd" />

아카라이브 채널 글과 이미지를 백업해 오프라인 HTML로 저장하는 프로젝트입니다.

## 실행 방법
```bash
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
python run.py  # http://127.0.0.1:5150
```

## 사용법
- 메인 페이지에서 채널, 카테고리, 페이지 범위, 저장 폴더명을 입력 후 **백업 시작**을 누르면 서버에 요청합니다.
- **자동 탐색**: '마지막 페이지'를 비워두면 채널의 끝까지 자동으로 탐색하여 백업합니다.
- **증분 백업**: 이미 백업된 폴더를 대상으로 다시 실행할 경우, `_listing.json`을 참조하여 신규 게시물만 탐색하고 백업을 진행합니다.
- **설정 불러오기**: 하단 "최근 백업 목록"에서 **설정 불러오기** 버튼을 클릭하면 해당 백업 당시의 옵션(채널, 카테고리 등)이 폼에 즉시 입력됩니다.
- 서버가 아카라이브를 통해 글/이미지를 내려받고, `backups/<폴더명>`에 각 글별 JSON과 오프라인 HTML을 만듭니다.
- 실행 결과는 페이지 우측 패널에 즉시 표시되고, 아래의 백업 기록 목록도 함께 갱신됩니다.

## 주요 파일
- `run.py`: Flask 앱 실행 스크립트.
- `app/__init__.py`: 앱 팩토리, 공통 설정 및 Jinja 필터 정의.
- `app/routes.py`: 백업 실행 엔드포인트 및 대시보드 라우트.
- `app/services/arca_backup.py`: 아카라이브 백업 로직(이미지 다운로드, manifest 생성 포함).
- `app/templates/`: 베이스 템플릿, 메인 페이지, 결과/이력 조각 템플릿.
- `app/static/`: Tailwind 번들 및 HTMX/Alpine 헬퍼 스크립트.

## 참고(차단 대응)
- 목록/게시글 모두 Playwright로 수집합니다. 필요하면 `PLAYWRIGHT_HEADLESS=0` 환경변수로 브라우저 창을 띄워 디버깅할 수 있습니다.
- 쿠키/로그인이 필요한 경우 `PLAYWRIGHT_STORAGE_STATE=/path/to/state.json` 으로 스토리지 상태를 주입할 수 있습니다.
