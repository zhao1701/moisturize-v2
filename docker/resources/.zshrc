# Path to your oh-my-zsh installation
export ZSH=/root/.oh-my-zsh

# Set name of theme to load
ZSH_THEME="avit"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
HYPHEN_INSENSITIVE="true"

# Uncomment the following line to enable command auto-correction.
ENABLE_CORRECTION="true"

plugins=(
  git
  pip
  python
  screen
  sublime
  sudo
  ubuntu
  vi-mode
  zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# Combine mkdir and cd
function mkcdir {
  mkdir $1
  cd $1
}

# Shortcuts
alias jpn='jupyter notebook'
