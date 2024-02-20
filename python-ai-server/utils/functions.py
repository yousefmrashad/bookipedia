from .config import *

# -- Main -- #

# Run python file
def run_python(file_name, args=[]):
    python_file = [os.path.join(ROOT, APP, file_name)]
    subprocess.Popen(START + python_file + args, shell=True)

# season & episodes config
def get_episodes(episodes: str):
    if (episodes):
        ep = episodes.split("-")
        
        if (len(ep) == 1):
            s = int(ep[0])-1
            return s, s+1
        else:
            s = int(ep[0])-1 if (ep[0]) else None
            e = int(ep[1])-0 if (ep[1]) else None

        return s, e
    else:
        return (None, None)
# ------------------------------------------------- #

# pre-downloading init files
def init_files(staged_item: dict, dl_title):
    vtype = staged_item["type"]
    title = staged_item["title"]
    season = staged_item["season"]
    uid = staged_item[dl_title].split("/")[4][:5]

    if (vtype == "Movie"):
        path = os.path.join(MOVIES_DIR, title)
    elif (vtype == "TV"):
        show_path = os.path.join(TVSHOWS_DIR, title)
        if not (os.path.isdir(show_path)): os.mkdir(show_path)
        path = os.path.join(TVSHOWS_DIR, title, f"Season {season}")
    
    tmp_path = os.path.join(path, uid)
    tmp_exists = os.path.isdir(tmp_path)

    if not (os.path.isdir(path)): os.mkdir(path)
    if not (tmp_exists): os.mkdir(tmp_path)
    
    return path, tmp_path, tmp_exists
# ------------------------------------------------- #

# unstage downloaded item
def unstage(vid, dl_title):
    with open(JSON_STAGED, "r+") as f:
        staged_data = json.loads(f.read())
    
        item = staged_data[vid]
        if (len(item.keys()) > 3):
            item.pop(dl_title)
        
        if (len(item.keys()) == 3):
            staged_data.pop(vid)
        
        f.seek(0)
        json.dump(staged_data, f)
        f.truncate()
# ------------------------------------------------- #

# -- Capture & Stage -- #

# minimize chrome
def minimize_chrome(w=0.5):
    sleep(w)
    while True:
        chrome_windows = pygetwindow.getWindowsWithTitle("Google Chrome")
        if (chrome_windows):
            chrome_windows[-1].minimize()
            break

# maximize chrome
def maximize_close_chrome():
    chrome_windows = pygetwindow.getWindowsWithTitle("Google Chrome")
    if (chrome_windows):
        chrome_window = chrome_windows[-1]
        chrome_window.maximize()
        chrome_window.close()
# ------------------------------------------- #

# style functions
def show_step(msg: str, a="-", c="white"):
    msg = f" {a} {msg.capitalize()}"
    print(colored(msg, c))

def show_title(msg: str, a="=", n=35, nl=True, c="white"):
    s = (116 - len(msg)) // 2
    print("\r", " "*LPAD)
    msg = f"{(a*n).rjust(s)} {msg} {a*n}"
    print(colored(msg, c))
    if (nl): print()

def quit(t=5, c="white"):
    for i in range(t):
        sys.stdout.write(colored(f"\r Quit in {t-i}s", c))
        sleep(1)
    exit()
# ------------------------------------------- #

# Filter Folders and File Names
def filter_name(name: str):
    return "".join(filter(lambda c: c not in SHIT_CHARS, name))
# ------------------------------------------- #