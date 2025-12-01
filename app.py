import os
import sys

# Ensure root path is accessible
sys.path.append(os.path.dirname(__file__))

# Directly run the Streamlit dashboard
import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/dashboard.py"]
    sys.exit(stcli.main())
