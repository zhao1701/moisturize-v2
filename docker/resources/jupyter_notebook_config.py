import os

c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 8000))

c.NotebookApp.allow_remote_access = True
c.NotebookApp.open_browser = False
c.NotebookApp.password = 'sha1:d40bf1265ca0:780f2f5b7d17f20efdc5b989bbbc82aa15d34609'

c.NotebookApp.notebook_dir = '/project/'
c.NotebookApp.allow_root = True
