FROM jjanzic/docker-python3-opencv:contrib

RUN git clone https://github.com/sirebellum/RPi-Edge /RPi-Edge

#Image in
EXPOSE 6001
#Boxes out
Expose 6002

WORKDIR "/RPi-Edge"

CMD ["python", "/RPi-Edge/main.py", "6", "2", "n"]
