# vistalab/qa_report_fmri
#
# Use modified CNI/NIMS code from @rfdougherty to create a qa_report for a given fmri NIfTI file in Flywheel spec.
# See https://github.com/cni/nims/blob/master/nimsproc/qa_report.py for original source code.
#
# Example usage:
#   docker run --rm -ti \
#        -v /path/nifti_file:/flywheel/v0/input \
#        -v /path/for/output/files:/flywheel/v0/output
#        scitran/qa-report-fmri /flywheel/v0/input -i nifti_file.nii.gz
#

FROM ubuntu-debootstrap:trusty

MAINTAINER Michael Perry <lmperry@stanford.edu>

# Install dependencies
RUN apt-get update && apt-get -y install \
    python-dev \
    python-pip \
    git \
    libjpeg-dev \
    zlib1g-dev \
    pkg-config \
    libpng12-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    zip \
    unzip

# Link libs: pillow jpegi and zlib support hack
RUN ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libz.so /usr/lib

# Install scitran.data dependencies
RUN  pip install pip==8.1.2 \
    && pip install numpy==1.11.0 \
    && pip install scipy==0.17.1 \
    && pip install dipy==0.11.0 \
    && pip install nibabel==2.0.2 \
    && pip install nipy==0.4.0 \
    && pip install matplotlib==1.5.1

# Trigger build of font cache
RUN python -c "from matplotlib import font_manager"

# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY run ${FLYWHEEL}/run
COPY manifest.json ${FLYWHEEL}/manifest.json

# Put the python code in place
COPY qa-report-fmri.py ${FLYWHEEL}/qa_report.py

ENTRYPOINT ["/flywheel/v0/run"]

