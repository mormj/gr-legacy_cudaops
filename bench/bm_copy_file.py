#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.0.0-git

from gnuradio import gr, blocks, analog
import sys
import signal
from argparse import ArgumentParser
import time
import legacy_cudaops
import bench
import os

class generate_file(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Generate File", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args.samples
        filename = args.filename

        ##################################################
        # Blocks
        ##################################################
        hd = blocks.head(gr.sizeof_gr_complex*1, 100000000)
       
        src = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 1, 0, 8192)

        snk = blocks.file_sink(gr.sizeof_gr_complex*1, filename, False)
        snk.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((src, 0), (hd, 0), (snk,0))


class benchmark_copy(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args.samples
        veclen = args.veclen
        load = args.load
        actual_samples = (
            veclen) * int(nsamples / veclen)
        num_blocks = args.nblocks
        mem_model = args.memmodel
        filename = args.filename

        ##################################################
        # Blocks
        ##################################################
        ptblocks = []
        for i in range(num_blocks):
            ptblocks.append(
                legacy_cudaops.copy(
                    veclen, load, mem_model)
            )

        src = blocks.file_source(gr.sizeof_gr_complex*1, filename, False, 0, 0)
        snk = blocks.file_sink(gr.sizeof_gr_complex*1, os.path.join(os.path.dirname(os.path.abspath(filename)),'outfile.dat'), False)
        snk.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((src, 0), (ptblocks[0], 0))

        for i in range(1, num_blocks):
            self.connect((ptblocks[i-1], 0), (ptblocks[i], 0))

        self.connect((ptblocks[num_blocks-1], 0),
                     (snk, 0))


def main(top_block_cls=benchmark_copy, options=None):
    parser = ArgumentParser(description='Run a flowgraph iterating over parameters for benchmarking')
    parser.add_argument('--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--samples', type=int, default=1e6)
    parser.add_argument('--veclen', type=int, default=1024)
    parser.add_argument('--nblocks', type=int, default=1)
    parser.add_argument('--load', type=int, default=1)
    parser.add_argument('--memmodel', type=int, default=0)


    args = parser.parse_args()
    print(args)

    if args.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    args.filename = os.path.join('/tmp',f'bm_copy_file_{int(args.samples)}')
    if not os.path.exists(args.filename):
        tb_pre = generate_file(args)
        tb_pre.run()

    tb = top_block_cls(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("starting ...")
    startt = time.time()
    tb.start()

    tb.wait()
    endt = time.time()

    # print(f'[PROFILE_VALID]{tb.snk.valid()}[PROFILE_VALID]')
    print(f'[PROFILE_TIME]{endt-startt}[PROFILE_TIME]')

if __name__ == '__main__':
    main()