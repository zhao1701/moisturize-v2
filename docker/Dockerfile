FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Derek Zhao (zhao1701@gmail.com)

# Basics
RUN apt-get update && apt-get install -y \
    vim git wget curl

# Python libraries
RUN pip install --upgrade pip
RUN pip install Pillow jupyter pandas scikit-learn matplotlib seaborn \
    keras pytest

# Notebook extensions
RUN pip install jupyter_contrib_nbextensions && \
    cd /usr/local/share/jupyter/nbextensions/ && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
    chmod -R go-w vim_binding && \
    jupyter contrib nbextension install --system && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable help_panel/help_panel && \
    jupyter nbextension enable highlight_selected_word/main && \
    jupyter nbextension enable keyboard_shortcut_editor/main && \
    jupyter nbextension enable move_selected_cells/main && \
    jupyter nbextension enable ruler/main && \
    jupyter nbextension enable scratchpad/main && \
    jupyter nbextension enable snippets/main && \
    jupyter nbextension enable toggle_all_line_numbers/main && \
    jupyter nbextension enable varInspector/main && \
    jupyter nbextension enable vim_binding/vim_binding && \
    jupyter nbextension enable zenmode/main

# Zsh
RUN apt-get install -y \
    zsh fonts-powerline
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Additional Python libraries
RUN pip install gitpython pillow tqdm imageio holoviews==1.12.1 bokeh pyviz
RUN pip install requests scikit-image
RUN pip install tensorflowjs

# Ports
EXPOSE 6006 8888

COPY resources/jupyter_notebook_config.py /root/.jupyter/
COPY resources/.vimrc /root/
COPY resources/custom.js /root/.jupyter/custom/custom.js
COPY resources/vim_binding.js /usr/local/share/jupyter/nbextensions/vim_binding/
COPY resources/.zshrc /root/
COPY resources/snippets.json /usr/local/share/jupyter/nbextensions/snippets

# ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat

WORKDIR /project/
CMD jupyter notebook
