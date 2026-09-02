from glob import glob

from setuptools import find_packages
from setuptools import setup


package_name = "beacon_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (
            f"share/{package_name}/config",
            glob("config/*.yaml") + glob("config/*.xml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "uwb_range_node = beacon_tracker.uwb_range_node:main",
            "packet_lock_node = beacon_tracker.packet_lock_node:main",
            "doa_angle_node = beacon_tracker.doa_angle_node:main",
            "beacon_monitor_node = beacon_tracker.beacon_monitor_node:main",
            "map_evidence_search_node = beacon_tracker.map_evidence_search_node:main",
            "evidence_diagnostics_logger_node = beacon_tracker.evidence_diagnostics_logger_node:main",
            "room_probability_monitor_node = beacon_tracker.room_probability_monitor_node:main",
            "camera_jpeg_relay_node = beacon_tracker.camera_jpeg_relay_node:main",
            "person_localizer_node = beacon_tracker.person_localizer_node:main",
            "jetson_yolo_inference_node = beacon_tracker.jetson_yolo_inference_node:main",
        ],
    },
)
