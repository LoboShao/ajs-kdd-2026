############################################################################
#
# Load Sharing Applications
#
# Makefile for AJS scheduler plugin
#
# Build:
#     all:   build all objects
#     clear clean: remove all object files
#
###########################################################################


#################### Testing purpose: by Yiming Shao#######################

# INCDIR = /opt/ibm/lsf/10.1/include
# LIBDIR = /opt/ibm/lsf/10.1/linux3.10-glibc2.17-x86_64/lib


###########################################################################


TOP= ../..
include $(TOP)/Make.misc

# LSF libraries needed

# Compiling flags
INCLUDEPATH = -I../../../include/lsf
CFLAGS=${EXTERN_CFLAGS} ${INCLUDEPATH} ${SITE} ${CFLAG_SHLIB}

#####################  Object files #########################
AJSOBJS = ajs.$(OEXT)

PLUGINAJS = schmod_ajs.$(SOEXT)

# build
build all: $(PLUGINAJS)


# build objects
.c.$(OEXT):   $<
	${CC} ${CFLAGS} ${SHLIB_CFLAGS} -c -o $@ $<

.c.$(STATIC_OEXT):   $<
	${CC} -DSTATIC_LINK ${CFLAGS} -c -o $@ $<

.c.i:  $<
	${CC} ${CFLAGS} -C -E -c -o $@ $<

###################  Binary & Release #######################
${PLUGINAJS}: ${AJSOBJS}
	$(SHLD) ${AJSOBJS} -lcurl -o ${PLUGINAJS}

# clean up all object files and temporary files
clean clear:
	-rm -f *.$(OEXT) *.$(SOEXT) .\#* 

# Optional: install target
install: $(PLUGINAJS)
	@echo "Installing plugin to LSF plugin directory..."
	# cp $(PLUGINAJS) $(LSF_LIBDIR)/plugins/
	@echo "Plugin built successfully: $(PLUGINAJS)"

# Optional: debug build
debug: CFLAGS += -g -DDEBUG
debug: $(PLUGINAJS)

.PHONY: build all clean clear install debug