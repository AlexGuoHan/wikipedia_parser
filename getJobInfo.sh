#! /bin/bash
jobid=`cat jobid.txt | cut -c21-28`
scontrol show jobid $jobid
~
