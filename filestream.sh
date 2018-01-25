IP=$1
FILE=$2
SLEEP=$3

if [ -z $1 ]
then
    echo "Please specify an IP!"
    exit
fi

if [ -z $2 ]
then
    echo "Please specify a h264 file!"
    exit
fi

if [ $SLEEP = 'y' ]
then sleep 2
fi

gst-launch-1.0 -v filesrc location=$FILE ! qtdemux ! video/x-h264 ! h264parse ! rtph264pay config-interval=1 ! udpsink host=$IP port=9001
