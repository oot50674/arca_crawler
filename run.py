import logging
from app import create_app


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


_configure_logging()
app = create_app()


if __name__ == '__main__':
    logging.info("Arca crawler running on http://127.0.0.1:5150")
    app.run(debug=True, port=5150)

