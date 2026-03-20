// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TinyCortex",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "TinyCortexEngine", targets: ["TinyCortexEngine"]),
        .executable(name: "TinyCortexApp", targets: ["TinyCortexApp"]),
    ],
    targets: [
        .target(
            name: "TinyCortexEngine",
            resources: [.process("Shaders")]
        ),
        .executableTarget(
            name: "TinyCortexApp",
            dependencies: ["TinyCortexEngine"],
            resources: [.copy("Resources")]
        ),
        .testTarget(
            name: "TinyCortexEngineTests",
            dependencies: ["TinyCortexEngine"],
            resources: [.copy("TestData")]
        ),
    ]
)
