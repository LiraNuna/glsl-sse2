#---------------------------------------------------------------------------------
.SUFFIXES:
.SILENT:

#---------------------------------------------------------------------------------
# TARGET is the name of the output
# EXT is the extension of the application (example: .exe)
# BUILD is the directory where object files & intermediate files will be placed
# SOURCES is a list of directories containing source code
# PACKAGES is a list of packages to link to the project (example: freefont)
# CROSS is a target for cross compilation ended with a dash (example: mingw32-msvc-)
# VERSION is GCC's version (example -3.4)
#---------------------------------------------------------------------------------
TARGET		:=	$(shell basename $(CURDIR))
EXT			:=	
BUILD		:=	.build
SOURCES		:=	source source/tests
PACKAGES	:=	
CROSS		:=	
VERSION		:=	

#---------------------------------------------------------------------------------
# options for code generation
#---------------------------------------------------------------------------------
ASFLAGS		:=	
CFLAGS		:=	-W -Wall -msse2 -g -O2
CXXFLAGS	:=	-fno-rtti -fno-exceptions -fomit-frame-pointer -ffast-math
LDFLAGS		:=	

#---------------------------------------------------------------------------------
# any extra libraries we wish to link with the project
#---------------------------------------------------------------------------------
LIBS		:=	

#---------------------------------------------------------------------------------
# everything is automatic from here on
#---------------------------------------------------------------------------------
CFLAGS		+=	$(INCLUDE) $(foreach pkg,$(PACKAGES),`pkg-config --cflags $(pkg)`)
LIBS		+=	$(foreach pkg,$(PACKAGES),`pkg-config --libs $(pkg)`)
CXXFLAGS	+=	$(CFLAGS)
LDFLAGS		+=	$(CFLAGS)

export AS	:=	$(CROSS)as$(VERSION)
export CC	:=	$(CROSS)gcc$(VERSION)
export CXX	:=	$(CROSS)g++$(VERSION)

#---------------------------------------------------------------------------------
%.o: %.cpp
	@echo $(notdir $<)
	$(CXX) -MMD -MP -MF $(DEPSDIR)/$*.d $(CXXFLAGS) -c $< -o $@
	
#---------------------------------------------------------------------------------
%.o: %.c
	@echo $(notdir $<)
	$(CC) -MMD -MP -MF $(DEPSDIR)/$*.d $(CFLAGS) -c $< -o $@

#---------------------------------------------------------------------------------
%.o: %.s
	@echo $(notdir $<)
	$(AS) --MD $(DEPSDIR)/$*.d $(ASFLAGS) $< -o$@

#---------------------------------------------------------------------------------
ifneq ($(BUILD),$(notdir $(CURDIR)))
#---------------------------------------------------------------------------------

export OUTPUT	:=	$(CURDIR)/$(TARGET)$(EXT)
export VPATH	:=	$(foreach dir,$(SOURCES),$(CURDIR)/$(dir)) \
					$(foreach dir,$(DATA),$(CURDIR)/$(dir))
export DEPSDIR	:=	$(CURDIR)/$(BUILD)

ASFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))
CFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))

#---------------------------------------------------------------------------------
# use CXX for linking C++ projects, CC for standard C
#---------------------------------------------------------------------------------
ifeq ($(strip $(CPPFILES)),)
	export LD	:=	$(CC)
else
	export LD	:=	$(CXX)
endif
#---------------------------------------------------------------------------------

export OFILES	:=	$(CPPFILES:.cpp=.o) $(CFILES:.c=.o)
export INCLUDE	:=	$(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
					$(foreach dir,$(LIBDIRS),-I$(dir)/include) \
					-I$(CURDIR)/$(BUILD)
export LIBPATHS	:=	$(foreach dir,$(LIBDIRS),-L$(dir)/lib)

.PHONY: $(BUILD) clean all Makefile

#---------------------------------------------------------------------------------
all: $(BUILD)

$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

#---------------------------------------------------------------------------------
clean:
	@echo clean ...
	$(RM) -rf $(BUILD) $(OUTPUT)

else

DEPENDS	:=	$(OFILES:.o=.d)

#---------------------------------------------------------------------------------
# main target
#---------------------------------------------------------------------------------
$(OUTPUT)	:	$(OFILES)
	@echo linking...
	@$(LD)  $(LDFLAGS) $(OFILES) $(LIBPATHS) $(LIBS) -o $@
	@strip $@
	@echo built $(notdir $@)

-include $(DEPENDS)

endif
