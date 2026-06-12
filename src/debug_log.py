import json
import os
import time

_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'debug-3ebdfd.log')


def agent_log(location, message, data=None, hypothesis_id=None, run_id='pre-fix'):
    # #region agent log
    entry = {
        'sessionId': '3ebdfd',
        'runId': run_id,
        'hypothesisId': hypothesis_id,
        'location': location,
        'message': message,
        'data': data or {},
        'timestamp': int(time.time() * 1000),
    }
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError:
        pass
    # #endregion
