FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# 기본 도구 + Python3/Pip 설치
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    zsh \
    vim \
    curl \
    locales \
    sudo \
    ca-certificates \
    tree \
    clang-format \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# 'python' 명령어를 'python3'로 연결
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# locale 설정
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# 비root 사용자 생성
RUN useradd -ms /bin/zsh user && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 사용자 전환
USER user
WORKDIR /home/user

# oh-my-zsh 설치
RUN sh -c "$(curl -fsSL https://gitee.com/mcornella/ohmyzsh/raw/master/tools/install.sh)" "" --unattended

# zsh 플러그인 설치
RUN git clone https://github.com/zsh-users/zsh-autosuggestions.git ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

# .zshrc 수정
RUN sed -i 's/ZSH_THEME=".*"/ZSH_THEME="robbyrussell"/' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# 기본 셸을 zsh로 설정
RUN sudo chsh -s $(which zsh) user

WORKDIR /workspace

# 시작 시 zsh 실행
CMD ["zsh"]
