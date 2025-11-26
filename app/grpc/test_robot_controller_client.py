#!/usr/bin/env python3
import grpc

import bero_pb2, bero_pb2_grpc

# 로봇 WARP 주소 + RobotController 포트
ROBOT_ADDR = "100.76.44.116:50051"


def main():
    print(f"[Client] RobotController에 접속 시도: {ROBOT_ADDR}")
    with grpc.insecure_channel(ROBOT_ADDR) as channel:
        stub = bero_pb2_grpc.RobotControllerStub(channel)

        # 1) Navigate 테스트
        nav_req = bero_pb2.NavigateRequest(waypoint="kitchen")
        nav_res = stub.Navigate(nav_req)
        print(f"[Client] Navigate 응답: success={nav_res.success}, message={nav_res.message}")

        # 2) Deliver 테스트
        del_req = bero_pb2.DeliverRequest(message="테스트 배달입니다.")
        del_res = stub.Deliver(del_req)
        print(f"[Client] Deliver 응답: success={del_res.success}")

        # 3) DescribeScene 테스트
        scene_req = bero_pb2.SceneRequest(prompt="지금 주변 상황 설명해줘")
        scene_res = stub.DescribeScene(scene_req)
        print(
            f"[Client] DescribeScene 응답: success={scene_res.success}, "
            f"description={scene_res.description}"
        )


if __name__ == "__main__":
    main()
