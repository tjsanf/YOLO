FROM tensorflow/tensorflow:1.15.2-gpu

RUN pip install --upgrade pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip install opencv-python
RUN pip install Pillow
RUN python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose