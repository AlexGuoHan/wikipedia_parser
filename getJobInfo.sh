#! /bin/sh


getJobInfos()
{
	task=$1
	jobid=`cat jobid.txt | cut -c21-28`	

	case $task in 
		now)
			scontrol show jobid $jobid
			;;
		past)
			sacct -j $jobid
			;;
		*) echo 'now or past'
			;;
	esac
}	

getJobInfos $@

