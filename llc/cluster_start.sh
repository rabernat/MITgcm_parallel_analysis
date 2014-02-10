# first created a profile
# http://ipython.org/ipython-doc/dev/parallel/parallel_process.html
# ipython profile create --parallel --profile=myprofile
# (only has to be done once)
# Then I edited it to set the working directory
# config files in ~/.ipython/profile_mpi/ipcluster_config.py
# then start it up with this command
ipcluster start --profile=mpi --n=256
# num procs should match the PBS job
