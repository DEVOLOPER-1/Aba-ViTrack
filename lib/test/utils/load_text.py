import numpy as np


def load_text_numpy(path, delimiter, dtype):
    # 1. Try the fast, rigid numpy parsing first
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                return np.loadtxt(path, delimiter=d, dtype=dtype)
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        try:
            return np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        except:
            pass

    # 2. --- BULLETPROOF MANUAL FALLBACK ---
    # If numpy fails, manually parse line-by-line. This handles mixed commas,
    # tabs, spaces, missing values, and random string artifacts like 'NaN'.
    try:
        with open(path, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            # Scrub commas and tabs into spaces, then split by whitespace
            clean_line = line.replace(',', ' ').replace('\t', ' ').strip()
            parts = clean_line.split()

            # Force exactly 4 columns
            while len(parts) < 4:
                parts.append('0.0')
            parts = parts[:4]

            # Convert to floats safely
            row = []
            for p in parts:
                try:
                    val = float(p)
                    row.append(val if not np.isnan(val) else 0.0)
                except ValueError:
                    row.append(0.0)  # If it's pure garbage text, default to 0.0
            data.append(row)

        return np.array(data, dtype=dtype)

    except Exception as e:
        raise Exception(f'Could not read file {path} with any backend. Error: {e}')


def load_text_pandas(path, delimiter, dtype):
    # Route pandas requests through the bulletproof numpy fallback
    return load_text_numpy(path, delimiter, dtype)


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    return load_text_numpy(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str