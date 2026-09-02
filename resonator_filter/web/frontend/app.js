let previousVictimCount = 0;

let victimInitialSyncDone = false;

// 같은 브라우저 탭에서는 새로고침해도
// 팝업이 다시 뜨지 않도록 저장
let victimPopupShown =
    sessionStorage.getItem("victimPopupShown") === "true";

/* =========================================================
   일반 사용자용 상태명
========================================================= */

const STATE_TEXT = {

    NAVIGATION:
        "목표 구역으로 이동 중",

    SEARCH_PATTERN:
        "비콘 탐색 중",

    ALIGNING:
        "비콘 방향 확인 중",

    LOCK_VERIFY:
        "비콘 신호 확인 중",

    TRACKING:
        "비콘 위치로 이동 중",

    PERSON_FOUND:
        "구조 대상 발견",

    WAITING:
        "대기 중",

    READY:
        "준비 중"
};


/* =========================================================
   WebSocket
========================================================= */

const protocol =
    location.protocol === "https:"
        ? "wss"
        : "ws";


const socket =
    new WebSocket(
        `${protocol}://${location.host}/ws`
    );


socket.onopen = function() {

    console.log(
        "Monitoring backend connected"
    );

};


socket.onerror = function(error) {

    console.error(
        "WebSocket error",
        error
    );

};


socket.onmessage = function(event) {

    const data =
        JSON.parse(
            event.data
        );


    updateRobot(data);

    updateGas(data);

    updateBeacon(data);

    updateVictims(data);

    updateEvents(data);

    updateMap(data);

};


/* =========================================================
   Robot
========================================================= */

function updateRobot(data) {

    const robot =
        data.robot;


    document.getElementById(
        "robot-state"
    ).innerText =

        STATE_TEXT[robot.state]
        ??
        robot.state
        ??
        "준비 중";


    /*
    로봇 위치가 아직 ROS에서 들어오지 않는 경우
    null 오류가 발생하지 않도록 처리
    */

    if (
        robot.x !== null
        &&
        robot.x !== undefined
        &&
        robot.y !== null
        &&
        robot.y !== undefined
    ) {

        document.getElementById(
            "robot-position"
        ).innerText =

            `(${Number(robot.x).toFixed(2)}, ${Number(robot.y).toFixed(2)})`;

    }

    else {

        document.getElementById(
            "robot-position"
        ).innerText =
            "위치 연결 대기";

    }


    if (
        robot.battery !== null
        &&
        robot.battery !== undefined
    ) {

        document.getElementById(
            "battery"
        ).innerText =

            `${Number(robot.battery).toFixed(1)}%`;

    }

    else {

        document.getElementById(
            "battery"
        ).innerText =
            "-";

    }

}


/* =========================================================
   Gas
========================================================= */

function updateGas(data) {

    const gas =
        data.gas;


    document.getElementById(
        "gas-value"
    ).innerText =

        gas.value === null
            ? "-"
            : gas.value;


    document.getElementById(
        "gas-unit"
    ).innerText =

        gas.unit
        ??
        "";


    document.getElementById(
        "gas-risk"
    ).innerText =

        gas.risk
        ??
        "센서 연결 대기";

}


/* =========================================================
   DOA 방향
========================================================= */

function directionText(angle) {

    if (
        angle === null
        ||
        angle === undefined
    ) {

        return "-";

    }


    if (
        Math.abs(angle) <= 5
    ) {

        return "정면";

    }


    if (
        angle > 0
    ) {

        return (
            `오른쪽 ${Math.abs(angle).toFixed(1)}°`
        );

    }


    return (
        `왼쪽 ${Math.abs(angle).toFixed(1)}°`
    );

}


/* =========================================================
   Beacon
========================================================= */

function updateBeacon(data) {

    const beacon =
        data.beacon;


    const status =
        document.getElementById(
            "beacon-status"
        );


    if (
        beacon.detected
    ) {

        status.innerText =
            "● 인식됨";

        status.className =
            "status-pill success";

    }

    else {

        status.innerText =
            "인식 불가";

        status.className =
            "status-pill neutral";

    }


    const packet =
        beacon.packet || {};

    const packetStatus =
        packet.status || "STARTING";

    const packetLocked =
        Boolean(packet.locked);

    document.getElementById(
        "packet-status"
    ).innerText =
        packetStatus;

    document.getElementById(
        "packet-lock"
    ).innerText =
        packetLocked
            ? "LOCKED"
            : "UNLOCKED";

    const level =
        packet.level_dbfs;

    const quality =
        packet.quality_db;

    const directionLevel =
        packet.direction_level_dbfs;

    const directionQuality =
        packet.direction_quality_db;

    document.getElementById(
        "packet-metric"
    ).innerText =
        Number.isFinite(level)
        && Number.isFinite(quality)
            ? level.toFixed(1) + " dBFS / " + quality.toFixed(1) + " dB"
            : "-";

    document.getElementById(
        "packet-direction-metric"
    ).innerText =
        Number.isFinite(directionLevel)
        && Number.isFinite(directionQuality)
            ? directionLevel.toFixed(1) + " dBFS / "
                + directionQuality.toFixed(1) + " dB"
            : "-";

    const direction =
        directionText(
            beacon.direction_deg
        );


    document.getElementById(
        "beacon-direction"
    ).innerText =
        direction;


    document.getElementById(
        "detail-direction"
    ).innerText =
        direction;


    document.getElementById(
        "tracking-status"
    ).innerText =

        beacon.tracking
            ? "진행 중"
            : "대기";


    const distance =
        data.uwb.distance_m;


    document.getElementById(
        "uwb-distance"
    ).innerText =

        distance === null
            ? "-"
            : `${Number(distance).toFixed(2)} m`;

}


/* =========================================================
   YOLO
========================================================= */

function updateYOLO(data) {

    const box =
        document.getElementById(
            "yolo-box"
        );


    if (
        data.yolo.detected
    ) {

        box.classList.remove(
            "hidden"
        );


        let label =
            "Person";


        if (
            data.yolo.confidence !== null
            &&
            data.yolo.confidence !== undefined
        ) {

            label +=
                ` ${data.yolo.confidence}%`;

        }


        document.getElementById(
            "yolo-label"
        ).innerText =
            label;

    }

    else {

        box.classList.add(
            "hidden"
        );

    }

}


function updateVictims(data) {

    const victims =
        data.victims
        ??
        [];


    // =========================
    // 사람 수 표시
    // =========================

    const victimCount =
        document.getElementById(
            "victim-count"
        );

    if (victimCount) {

        victimCount.innerText =
            victims.length;

    }


    // =========================
    // PERSON 태그 표시
    // =========================

    const list =
        document.getElementById(
            "victim-list"
        );

    if (list) {

        list.innerHTML = "";

        victims.forEach(person => {

            const card =
                document.createElement(
                    "div"
                );

            card.className =
                "victim-card";

            card.innerHTML =
                `
                <div class="victim-card-title">

                    <span class="person-dot"></span>

                    <strong>
                        ${person.id}
                    </strong>

                </div>
                `;

            list.appendChild(card);

        });

    }


    // =========================
    // 첫 WebSocket 연결
    //
    // 새로고침했을 때 기존 PERSON을
    // 새 발견으로 착각하지 않도록 함
    // =========================

if (!victimInitialSyncDone) {

    previousVictimCount =
        victims.length;

    victimInitialSyncDone =
        true;


    /*
     * 서버에 등록된 구조 대상이 0명이라면
     * 새로운 탐색이 시작된 것으로 판단.
     * 이전 임무의 팝업 기록을 초기화한다.
     */
    if (victims.length === 0) {

        victimPopupShown = false;

        sessionStorage.removeItem(
            "victimPopupShown"
        );
    }


    return;
}

    // =========================
    // 최초 발견 팝업 딱 1회
    // =========================

    if (
        !victimPopupShown
        &&
        previousVictimCount === 0
        &&
        victims.length > 0
    ) {

        showVictimPopup(
            victims[0]
        );

        victimPopupShown = true;

        sessionStorage.setItem(
            "victimPopupShown",
            "true"
        );

    }


    previousVictimCount =
        victims.length;
}

/* =========================================================
   Victim popup
========================================================= */

function showVictimPopup(person) {

    const modal =
        document.getElementById(
            "victim-modal"
        );


    const x =
        person.x !== null
            ? Number(person.x).toFixed(2)
            : "-";


    const y =
        person.y !== null
            ? Number(person.y).toFixed(2)
            : "-";


    document.getElementById(
        "modal-victim-data"
    ).innerHTML =

        `
        <div>
            <span>구조 대상</span>
            <strong>${person.id}</strong>
        </div>

        <div>
            <span>발견 위치</span>
            <strong>
                ${x},
                ${y}
            </strong>
        </div>

        <div>
            <span>발견 시간</span>
            <strong>
                ${person.time ?? "-"}
            </strong>
        </div>

        <div>
            <span>가스 값</span>
            <strong>
                ${person.gas ?? "-"}
            </strong>
        </div>
        `;


    modal.classList.remove(
        "hidden"
    );

}


/* =========================================================
   Events
========================================================= */

function updateEvents(data) {

    const container =
        document.getElementById(
            "recent-events"
        );


    container.innerHTML =
        "";


    const events =
        (
            data.events
            ??
            []
        ).slice(
            0,
            5
        );


    if (
        events.length === 0
    ) {

        container.innerHTML =
            `
            <div class="event-item">
                <span class="event-time">-</span>
                시스템이 준비 중입니다.
            </div>
            `;

        return;

    }


    events.forEach(
        event => {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "event-item";


            item.innerHTML =
                `
                <span class="event-time">
                    ${event.time}
                </span>

                <span>
                    ${event.message}
                </span>
                `;


            container.appendChild(
                item
            );

        }
    );

}


/* =========================================================
   INTERACTIVE ROS MAP
========================================================= */

let mapImage =
    null;


let loadedMapUrl =
    null;


let latestMapData =
    null;


/*
mapZoom

1 = 전체보기에 맞춘 기본 크기
2 = 2배 확대
...
*/

let mapZoom =
    1;


let mapPanX =
    0;


let mapPanY =
    0;


let mapDragging =
    false;


let mapDragStartX =
    0;


let mapDragStartY =
    0;


let mapPanStartX =
    0;


let mapPanStartY =
    0;


/* =========================================================
   지도 데이터 업데이트
========================================================= */

function updateMap(data) {

    latestMapData =
        data;


    const map =
        data.map;


    if (
        !map
        ||
        !map.connected
        ||
        !map.image_url
    ) {

        drawEmptyMap();

        return;

    }


    /*
    새로운 지도 URL이면 이미지 다시 로드

    map.yaml에서 읽은 실제 PGM 지도를
    backend의 /map_image를 통해 받음.
    */

    if (
        loadedMapUrl
        !==
        map.image_url
    ) {

        loadedMapUrl =
            map.image_url;


        mapImage =
            new Image();


        mapImage.onload =
            function() {

                resetMapView();

            };


        mapImage.onerror =
            function() {

                console.error(
                    "ROS map image load failed"
                );


                drawEmptyMap();

            };


        mapImage.src =
            map.image_url
            +
            `?v=${Date.now()}`;

    }

    else {

        drawInteractiveMap();

    }

}


/* =========================================================
   지도 전체보기
========================================================= */

function resetMapView() {

    mapZoom =
        1;


    mapPanX =
        0;


    mapPanY =
        0;


    drawInteractiveMap();

}


/* =========================================================
   ROS 좌표 → PGM 지도 픽셀 좌표

   YAML:
   resolution
   origin
   width
   height

   을 사용하기 때문에 맵이 바뀌어도
   새 YAML 값을 기준으로 자동 변환됨.
========================================================= */

function rosToMapPixel(
    x,
    y,
    map
) {

    const resolution =
        Number(
            map.resolution
        );


    const originX =
        Number(
            map.origin[0]
        );


    const originY =
        Number(
            map.origin[1]
        );


    const pixelX =
        (
            Number(x)
            -
            originX
        )
        /
        resolution;


    /*
    ROS map의 Y 방향과
    이미지의 Y 방향은 반대.
    */

    const pixelY =
        Number(map.height)
        -
        (
            (
                Number(y)
                -
                originY
            )
            /
            resolution
        );


    return {

        x:
            pixelX,

        y:
            pixelY

    };

}


/* =========================================================
   현재 지도 Transform 계산
========================================================= */

function getMapTransform() {

    if (
        !latestMapData
        ||
        !latestMapData.map
    ) {

        return null;

    }


    const canvas =
        document.getElementById(
            "map-canvas"
        );


    if (
        !canvas
    ) {

        return null;

    }


    const rect =
        canvas.getBoundingClientRect();


    const map =
        latestMapData.map;


    const mapWidth =
        Number(
            map.width
        );


    const mapHeight =
        Number(
            map.height
        );


    if (
        !mapWidth
        ||
        !mapHeight
        ||
        rect.width === 0
        ||
        rect.height === 0
    ) {

        return null;

    }


    /*
    기본 fit scale.

    PGM 전체가 화면 안에 최대한 크게 표시됨.
    */

    const fitScale =
        Math.min(

            rect.width
            /
            mapWidth,

            rect.height
            /
            mapHeight

        );


    const scale =
        fitScale
        *
        mapZoom;


    const displayWidth =
        mapWidth
        *
        scale;


    const displayHeight =
        mapHeight
        *
        scale;


    const baseX =
        (
            rect.width
            -
            displayWidth
        )
        /
        2;


    const baseY =
        (
            rect.height
            -
            displayHeight
        )
        /
        2;


    return {

        rect:
            rect,

        scale:
            scale,

        offsetX:
            baseX
            +
            mapPanX,

        offsetY:
            baseY
            +
            mapPanY,

        displayWidth:
            displayWidth,

        displayHeight:
            displayHeight

    };

}


/* =========================================================
   지도 픽셀 → 화면 Canvas 좌표
========================================================= */

function mapPixelToCanvas(
    point,
    transform
) {

    return {

        x:
            transform.offsetX
            +
            point.x
            *
            transform.scale,

        y:
            transform.offsetY
            +
            point.y
            *
            transform.scale

    };

}


/* =========================================================
   실제 지도 전체 그리기
========================================================= */

function drawInteractiveMap() {

    if (
        !latestMapData
        ||
        !mapImage
        ||
        !mapImage.complete
    ) {

        return;

    }


    const canvas =
        document.getElementById(
            "map-canvas"
        );


    if (
        !canvas
    ) {

        return;

    }


    const rect =
        canvas.getBoundingClientRect();


    if (
        rect.width === 0
        ||
        rect.height === 0
    ) {

        return;

    }


    /*
    Retina/고해상도 화면에서도
    지도와 글자가 흐리지 않도록 DPR 적용.
    */

    const dpr =
        window.devicePixelRatio
        ||
        1;


    canvas.width =
        Math.round(
            rect.width
            *
            dpr
        );


    canvas.height =
        Math.round(
            rect.height
            *
            dpr
        );


    const ctx =
        canvas.getContext(
            "2d"
        );


    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );


    ctx.clearRect(
        0,
        0,
        rect.width,
        rect.height
    );


    /*
    지도 바깥 영역
    */

    ctx.fillStyle =
        "#d1d1d1";


    ctx.fillRect(
        0,
        0,
        rect.width,
        rect.height
    );


    const transform =
        getMapTransform();


    if (
        !transform
    ) {

        return;

    }


    /*
    실제 ROS 지도
    */

    ctx.drawImage(

        mapImage,

        transform.offsetX,
        transform.offsetY,

        transform.displayWidth,
        transform.displayHeight

    );


    /*
    지도 위 Overlay
    */

    drawRobotPath(
        ctx,
        transform
    );


    drawVictimMarkers(
        ctx,
        transform
    );


    drawRobotMarker(
        ctx,
        transform
    );

}


/* =========================================================
   TurtleBot 이동 경로
========================================================= */

function drawRobotPath(
    ctx,
    transform
) {

    const path =
        latestMapData.path
        ??
        [];


    if (
        path.length < 2
    ) {

        return;

    }


    ctx.beginPath();


    path.forEach(
        (
            point,
            index
        ) => {

            if (
                point.x === null
                ||
                point.y === null
            ) {

                return;

            }


            const mapPoint =
                rosToMapPixel(
                    point.x,
                    point.y,
                    latestMapData.map
                );


            const p =
                mapPixelToCanvas(
                    mapPoint,
                    transform
                );


            if (
                index === 0
            ) {

                ctx.moveTo(
                    p.x,
                    p.y
                );

            }

            else {

                ctx.lineTo(
                    p.x,
                    p.y
                );

            }

        }
    );


    ctx.strokeStyle =
        "#0071e3";


    ctx.lineWidth =
        3;


    ctx.lineCap =
        "round";


    ctx.lineJoin =
        "round";


    ctx.stroke();

}


/* =========================================================
   구조 대상 마커
========================================================= */

function drawVictimMarkers(
    ctx,
    transform
) {

    const victims =
        latestMapData.victims
        ??
        [];


    victims.forEach(
        person => {

            if (
                person.x === null
                ||
                person.x === undefined
                ||
                person.y === null
                ||
                person.y === undefined
            ) {

                return;

            }


            const mapPoint =
                rosToMapPixel(
                    person.x,
                    person.y,
                    latestMapData.map
                );


            const p =
                mapPixelToCanvas(
                    mapPoint,
                    transform
                );


            /*
            빨간 구조 대상 마커
            */

            ctx.beginPath();


            ctx.arc(
                p.x,
                p.y,
                9,
                0,
                Math.PI * 2
            );


            ctx.fillStyle =
                "#d70015";


            ctx.fill();


            ctx.strokeStyle =
                "#ffffff";


            ctx.lineWidth =
                3;


            ctx.stroke();


            /*
            PERSON_01 라벨
            */

            ctx.font =
                "600 13px -apple-system, BlinkMacSystemFont, sans-serif";


            ctx.fillStyle =
                "#1d1d1f";


            ctx.fillText(
                person.id,
                p.x + 15,
                p.y + 5
            );

        }
    );

}


/* =========================================================
   TurtleBot 위치 마커
========================================================= */

function drawRobotMarker(
    ctx,
    transform
) {

    const robot =
        latestMapData.robot;


    if (
        !robot
        ||
        robot.x === null
        ||
        robot.x === undefined
        ||
        robot.y === null
        ||
        robot.y === undefined
    ) {

        return;

    }


    const mapPoint =
        rosToMapPixel(
            robot.x,
            robot.y,
            latestMapData.map
        );


    const p =
        mapPixelToCanvas(
            mapPoint,
            transform
        );


    /*
    로봇 파란 마커
    */

    ctx.beginPath();


    ctx.arc(
        p.x,
        p.y,
        11,
        0,
        Math.PI * 2
    );


    ctx.fillStyle =
        "#0071e3";


    ctx.fill();


    ctx.strokeStyle =
        "#ffffff";


    ctx.lineWidth =
        3;


    ctx.stroke();


    /*
    중심점
    */

    ctx.beginPath();


    ctx.arc(
        p.x,
        p.y,
        3,
        0,
        Math.PI * 2
    );


    ctx.fillStyle =
        "#ffffff";


    ctx.fill();


    /*
    로봇 방향
    */

    if (
        robot.yaw_deg !== null
        &&
        robot.yaw_deg !== undefined
    ) {

        /*
        ROS yaw는 반시계 방향 +
        Canvas Y축은 아래 방향 +

        따라서 화면에서는 부호 반전.
        */

        const angle =
            (
                -Number(
                    robot.yaw_deg
                )
            )
            *
            Math.PI
            /
            180;


        const length =
            28;


        ctx.beginPath();


        ctx.moveTo(
            p.x,
            p.y
        );


        ctx.lineTo(

            p.x
            +
            Math.cos(
                angle
            )
            *
            length,

            p.y
            +
            Math.sin(
                angle
            )
            *
            length

        );


        ctx.strokeStyle =
            "#0071e3";


        ctx.lineWidth =
            4;


        ctx.lineCap =
            "round";


        ctx.stroke();

    }

}


/* =========================================================
   지도 미연결 상태
========================================================= */

function drawEmptyMap() {

    const canvas =
        document.getElementById(
            "map-canvas"
        );


    if (
        !canvas
    ) {

        return;

    }


    const rect =
        canvas.getBoundingClientRect();


    const dpr =
        window.devicePixelRatio
        ||
        1;


    canvas.width =
        Math.round(
            rect.width
            *
            dpr
        );


    canvas.height =
        Math.round(
            rect.height
            *
            dpr
        );


    const ctx =
        canvas.getContext(
            "2d"
        );


    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );


    ctx.fillStyle =
        "#f5f5f7";


    ctx.fillRect(
        0,
        0,
        rect.width,
        rect.height
    );


    ctx.fillStyle =
        "#86868b";


    ctx.textAlign =
        "center";


    ctx.font =
        "14px -apple-system, BlinkMacSystemFont, sans-serif";


    ctx.fillText(
        "ROS 지도 연결 대기",
        rect.width / 2,
        rect.height / 2
    );

}


/* =========================================================
   지도 확대 / 축소
========================================================= */

function zoomMap(
    factor
) {

    const oldZoom =
        mapZoom;


    let newZoom =
        oldZoom
        *
        factor;


    /*
    최소 = 전체보기보다 조금 작게
    최대 = 8배 확대
    */

    newZoom =
        Math.max(
            0.8,
            Math.min(
                8,
                newZoom
            )
        );


    if (
        newZoom === oldZoom
    ) {

        return;

    }


    mapZoom =
        newZoom;


    drawInteractiveMap();

}


/* =========================================================
   Map Viewer Event
========================================================= */

const mapViewer =
    document.getElementById(
        "map-viewer"
    );


if (
    mapViewer
) {

    /*
    마우스 휠 확대 / 축소
    */

    mapViewer.addEventListener(

        "wheel",

        function(event) {

            event.preventDefault();


            if (
                event.deltaY < 0
            ) {

                zoomMap(
                    1.15
                );

            }

            else {

                zoomMap(
                    1 / 1.15
                );

            }

        },

        {
            passive:
                false
        }

    );


    /*
    지도 드래그 시작
    */

    mapViewer.addEventListener(

        "pointerdown",

        function(event) {

            mapDragging =
                true;


            mapViewer.classList.add(
                "dragging"
            );


            mapDragStartX =
                event.clientX;


            mapDragStartY =
                event.clientY;


            mapPanStartX =
                mapPanX;


            mapPanStartY =
                mapPanY;


            mapViewer.setPointerCapture(
                event.pointerId
            );

        }

    );


    /*
    지도 드래그 이동
    */

    mapViewer.addEventListener(

        "pointermove",

        function(event) {

            if (
                !mapDragging
            ) {

                return;

            }


            mapPanX =
                mapPanStartX
                +
                (
                    event.clientX
                    -
                    mapDragStartX
                );


            mapPanY =
                mapPanStartY
                +
                (
                    event.clientY
                    -
                    mapDragStartY
                );


            drawInteractiveMap();

        }

    );


    /*
    지도 드래그 종료
    */

    mapViewer.addEventListener(

        "pointerup",

        function() {

            mapDragging =
                false;


            mapViewer.classList.remove(
                "dragging"
            );

        }

    );


    mapViewer.addEventListener(

        "pointercancel",

        function() {

            mapDragging =
                false;


            mapViewer.classList.remove(
                "dragging"
            );

        }

    );

}


/* =========================================================
   지도 컨트롤 버튼
========================================================= */

const mapZoomInButton =
    document.getElementById(
        "map-zoom-in"
    );


const mapZoomOutButton =
    document.getElementById(
        "map-zoom-out"
    );


const mapResetButton =
    document.getElementById(
        "map-reset"
    );


if (
    mapZoomInButton
) {

    mapZoomInButton.onclick =
        function() {

            zoomMap(
                1.3
            );

        };

}


if (
    mapZoomOutButton
) {

    mapZoomOutButton.onclick =
        function() {

            zoomMap(
                1 / 1.3
            );

        };

}


if (
    mapResetButton
) {

    mapResetButton.onclick =
        function() {

            resetMapView();

        };

}


/* =========================================================
   구조 대상 마커 클릭
========================================================= */

const mapCanvas =
    document.getElementById(
        "map-canvas"
    );


if (
    mapCanvas
) {

    mapCanvas.addEventListener(

        "click",

        function(event) {

            /*
            드래그 직후 발생하는 click은
            마커 선택으로 처리하지 않음.
            */

            if (
                !latestMapData
            ) {

                return;

            }


            const victims =
                latestMapData.victims
                ??
                [];


            const rect =
                mapCanvas.getBoundingClientRect();


            const clickX =
                event.clientX
                -
                rect.left;


            const clickY =
                event.clientY
                -
                rect.top;


            const transform =
                getMapTransform();


            if (
                !transform
            ) {

                return;

            }


            for (
                const person
                of victims
            ) {

                if (
                    person.x === null
                    ||
                    person.y === null
                ) {

                    continue;

                }


                const mapPoint =
                    rosToMapPixel(
                        person.x,
                        person.y,
                        latestMapData.map
                    );


                const p =
                    mapPixelToCanvas(
                        mapPoint,
                        transform
                    );


                const distance =
                    Math.hypot(

                        clickX
                        -
                        p.x,

                        clickY
                        -
                        p.y

                    );


                if (
                    distance <= 20
                ) {

                    showMapVictimPopup(
                        person,
                        clickX,
                        clickY
                    );


                    return;

                }

            }


            hideMapPopup();

        }

    );

}


/* =========================================================
   지도 구조 대상 정보창
========================================================= */

function showMapVictimPopup(
    person,
    x,
    y
) {

    const popup =
        document.getElementById(
            "map-popup"
        );


    if (
        !popup
    ) {

        return;

    }


    popup.innerHTML =
        `
        <strong>
            ${person.id}
        </strong>

        <p>
            위치:
            ${Number(person.x).toFixed(2)},
            ${Number(person.y).toFixed(2)}
        </p>

        <p>
            발견 시간:
            ${person.time ?? "-"}
        </p>

        <p>
            가스:
            ${person.gas ?? "-"}
        </p>
        `;


    popup.classList.remove(
        "hidden"
    );


    /*
    map-popup을 map-viewer 내부 absolute로
    사용하기 때문에 canvas 좌표 그대로 사용.
    */

    popup.style.left =
        `${x + 14}px`;


    popup.style.top =
        `${y + 14}px`;

}


function hideMapPopup() {

    const popup =
        document.getElementById(
            "map-popup"
        );


    if (
        popup
    ) {

        popup.classList.add(
            "hidden"
        );

    }

}


/* =========================================================
   브라우저 크기 변경
========================================================= */

window.addEventListener(

    "resize",

    function() {

        drawInteractiveMap();

    }

);


/* =========================================================
   기존 Modal Buttons
========================================================= */

document.getElementById(
    "modal-close"
).onclick = function() {

    document.getElementById(
        "victim-modal"
    ).classList.add(
        "hidden"
    );

};


document.getElementById(
    "beacon-detail-button"
).onclick = function() {

    document.getElementById(
        "beacon-modal"
    ).classList.remove(
        "hidden"
    );

};


document.getElementById(
    "beacon-modal-close"
).onclick = function() {

    document.getElementById(
        "beacon-modal"
    ).classList.add(
        "hidden"
    );

};
