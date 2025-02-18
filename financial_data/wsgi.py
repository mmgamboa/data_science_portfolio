# wsgi.py
from outlier_returns import server as application

if __name__ == "__main__":
    application.run()