#!/bin/bash
# Install the guardian cron entry (every minute + at boot; flock keeps one copy; -o so children never inherit the lock) and start it now.
D=/root/guardian; chmod +x $D/*.sh $D/feed.py; mkdir -p $D/out $D/q
( crontab -l 2>/dev/null | grep -v 'guardian/run.sh'; echo "* * * * * /usr/bin/flock -n -o $D/run.lock $D/run.sh"; echo "@reboot /usr/bin/flock -n -o $D/run.lock $D/run.sh" ) | crontab -
pgrep cron > /dev/null || (cron 2>/dev/null || service cron start)
nohup /usr/bin/flock -n -o $D/run.lock $D/run.sh > /dev/null 2>&1 &
