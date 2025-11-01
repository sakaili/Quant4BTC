# main_binance.py
from main import main as run_main


def main():
    """Binance-specific entrypoint that delegates to the shared runner."""

    run_main()


if __name__ == "__main__":
    main()
