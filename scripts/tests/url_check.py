url = "https://infoveave-help.pages.dev/insights-v8/guide-to-infoboard-designer/customizie-panel/customization-tab/"

if not url.endswith("/"):
    url = url + "/" # Remove trailing slash if present

def update_url(url):
    if not url.endswith("/"):
        url += "/"
    return url