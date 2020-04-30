FROM alpine:edge
RUN apk add cmake make gcc g++ libc-dev linux-headers doxygen graphviz ttf-freefont
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]