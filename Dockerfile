# GNSS interference detector as a self-contained map server.
#
#   docker build -t gnss-detector .
#   docker run --rm -p 8080:8080 gnss-detector
#
# On start the container refreshes theater.geojson from GPSJam (falling back
# to the bundled snapshot if offline), labels hot cells — live if
# OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET are set, otherwise --demo — then
# rebuilds and serves the interactive map at http://localhost:8080/
FROM python:3.12-slim

# detect.py prints Unicode arrows; force UTF-8 regardless of container locale.
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

RUN pip install --no-cache-dir requests h3

COPY detect.py classify.py build_map.py theater.geojson labels.json \
     gnss-jamming-spoofing-map.html ./

CMD ["sh", "-c", "\
    (python detect.py || echo '[docker] GPSJam fetch failed; using bundled snapshot') && \
    if [ -n \"$OPENSKY_CLIENT_ID\" ]; then \
        python classify.py || echo '[docker] live classify failed; keeping bundled labels'; \
    else \
        python classify.py --demo || true; \
    fi && \
    python build_map.py && \
    python -m http.server 8080"]
