// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "splash-studio",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "splash-studio", path: "Sources"),
        .testTarget(name: "StudioTests", dependencies: ["splash-studio"],
                    path: "Tests", resources: [.copy("Fixtures")])
    ]
)
