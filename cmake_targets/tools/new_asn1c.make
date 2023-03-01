# this is a hack - it will be removed at some point
new_asn1c:
	echo asn1c prefix="$(INSTALL_DIR)"
	rm -rf "$(INSTALL_DIR)/src"
	git clone https://github.com/mouse07410/asn1c "$(INSTALL_DIR)/src"
	cd "$(INSTALL_DIR)/src" && \
	git checkout vlm_master && \
	autoreconf -iv && \
	./configure --prefix="$(INSTALL_DIR)" && \
	make -j`nproc` install
