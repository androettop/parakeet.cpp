#!/usr/bin/env python3
"""Tiny static file server for the parakeet.cpp WASM demo.

    python3 examples/wasm/serve.py [port]

Serves the examples/wasm/ directory at http://localhost:8000 (default). Sets the
correct application/wasm MIME type and, so the pthreads build (SharedArrayBuffer)
also works, the cross-origin isolation headers (COOP/COEP). The default
single-threaded build does not need them; they are harmless either way.
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable SharedArrayBuffer (needed only by the pthreads build).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


Handler.extensions_map[".wasm"] = "application/wasm"
Handler.extensions_map[".js"] = "text/javascript"
Handler.extensions_map[".mjs"] = "text/javascript"

if __name__ == "__main__":
    with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"parakeet.cpp WASM demo: http://localhost:{PORT}/  (serving {DIRECTORY})")
        print("Build the module first: scripts/build_wasm.sh")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
