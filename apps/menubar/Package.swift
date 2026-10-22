// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "splash-menubar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "splash-menubar", path: "Sources")
    ]
)
