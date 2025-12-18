import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright

def make_storage_state():
    state_path = "storage_state.json"
    print(f"브라우저를 실행합니다. 아카라이브(https://arca.live)에 접속합니다.")
    print(f"필요하다면 로그인을 진행해 주세요.")
    print(f"완료되면 브라우저를 닫거나, 이 터미널에서 Ctrl+C를 눌러 종료하면 {state_path}가 저장됩니다.")
    
    with sync_playwright() as p:
        # 사용자가 직접 조작할 수 있도록 headless=False로 실행
        # (참고: 원격 환경이나 GUI가 없는 환경에서는 작동하지 않을 수 있습니다)
        try:
            browser = p.chromium.launch(headless=False)
        except Exception as e:
            print(f"GUI 브라우저 실행 실패: {e}")
            print("headless 모드로 전환하여 기본 쿠키만 수집합니다.")
            browser = p.chromium.launch(headless=True)
            
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            page.goto("https://arca.live", wait_until="networkidle")
            
            # 브라우저가 닫힐 때까지 대기
            print("\n>>> 브라우저에서 작업을 마치고 브라우저 창을 닫아주세요. <<<")
            page.wait_for_event("close", timeout=0)
        except KeyboardInterrupt:
            print("\n중단 요청을 감지했습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
        finally:
            context.storage_state(path=state_path)
            print(f"\n성공: {state_path} 파일이 생성되었습니다.")
            browser.close()

if __name__ == "__main__":
    make_storage_state()
