ARG OPENWEBUI_BASE_IMAGE=ghcr.io/open-webui/open-webui:main

FROM node:22-alpine3.20 AS build

ARG OPENWEBUI_GIT_REPO
ARG OPENWEBUI_GIT_REF=main
ARG VITE_GPTHUB_VISION_MODELS=""
ARG VITE_GPTHUB_AUDIO_MODELS=""

ENV NODE_OPTIONS="--max-old-space-size=6144"
ENV CYPRESS_INSTALL_BINARY=0

WORKDIR /src

RUN apk add --no-cache git

RUN git clone --depth 1 --branch "${OPENWEBUI_GIT_REF}" "${OPENWEBUI_GIT_REPO}" openwebui || \
	(git clone "${OPENWEBUI_GIT_REPO}" openwebui && cd openwebui && git checkout "${OPENWEBUI_GIT_REF}")

WORKDIR /src/openwebui

RUN npm config set fetch-retries 5 \
	&& npm config set fetch-retry-factor 2 \
	&& npm config set fetch-retry-mintimeout 20000 \
	&& npm config set fetch-retry-maxtimeout 120000 \
	&& npm config set fetch-timeout 600000 \
	&& npm config set audit false \
	&& npm config set fund false \
	&& retry=0 \
	&& until [ "$retry" -ge 3 ]; do \
		npm ci --force && break; \
		retry=$((retry + 1)); \
		echo "npm ci failed, retry ${retry}/3"; \
		npm cache clean --force; \
		sleep $((retry * 10)); \
	done \
	&& [ "$retry" -lt 3 ]

ENV APP_BUILD_HASH=${OPENWEBUI_GIT_REF}
ENV VITE_GPTHUB_VISION_MODELS=${VITE_GPTHUB_VISION_MODELS}
ENV VITE_GPTHUB_AUDIO_MODELS=${VITE_GPTHUB_AUDIO_MODELS}

RUN npm run build

FROM ${OPENWEBUI_BASE_IMAGE}

ARG OPENWEBUI_GIT_REPO
ARG OPENWEBUI_GIT_REF=main
ARG OPENWEBUI_BASE_IMAGE

LABEL org.opencontainers.image.source="${OPENWEBUI_GIT_REPO}"
LABEL org.opencontainers.image.revision="${OPENWEBUI_GIT_REF}"
LABEL org.opencontainers.image.base.name="${OPENWEBUI_BASE_IMAGE}"

COPY --from=build /src/openwebui/build /app/build
COPY --from=build /src/openwebui/backend /app/backend
COPY --from=build /src/openwebui/CHANGELOG.md /app/CHANGELOG.md
COPY --from=build /src/openwebui/package.json /app/package.json
