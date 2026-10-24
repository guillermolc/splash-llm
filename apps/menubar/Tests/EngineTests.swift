import XCTest
import Foundation
import Darwin
@testable import splash_studio

final class EngineTests: XCTestCase {
    @MainActor
    func testOwnedStartIdleRestartAndStop() async throws {
        try await withFixture { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .running }
            XCTAssertEqual(engine.ownership, .owned)
            XCTAssertEqual(engine.appliedSettings, fixture.settings)
            let previousPID = try XCTUnwrap(engine.snapshot?.pid)
            var changed = fixture.settings
            changed.context = "64K"
            try await engine.restart(root: fixture.root.path, model: "fixture/base", s: changed)
            try await waitUntil { engine.state == .running }
            XCTAssertNotEqual(engine.snapshot?.pid, previousPID)
            XCTAssertFalse(isAlive(previousPID))
            XCTAssertEqual(engine.appliedSettings?.context, "64K")
            await engine.stop()
            XCTAssertEqual(engine.state, .stopped)
            XCTAssertEqual(engine.ownership, .none)
            XCTAssertNil(engine.snapshot)
            let available = await ProcTree.portAvailable(fixture.port)
            XCTAssertTrue(available)
        }
    }

    @MainActor
    func testStopDuringStartupCannotPublishLateState() async throws {
        try await withFixture(control: ["startup_delay": 2]) { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            XCTAssertEqual(engine.state, .starting)
            await engine.stop()
            try await Task.sleep(nanoseconds: 2_100_000_000)
            XCTAssertEqual(engine.state, .stopped)
            XCTAssertEqual(engine.ownership, .none)
            XCTAssertNil(engine.snapshot)
            let available = await ProcTree.portAvailable(fixture.port)
            XCTAssertTrue(available)
        }
    }

    @MainActor
    func testRestartPreflightFailurePreservesTheWorkingInstance() async throws {
        try await withFixture { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .running }
            let pid = try XCTUnwrap(engine.snapshot?.pid)
            var missing = fixture.settings
            missing.model = "fixture/missing"
            do {
                try await engine.restart(root: fixture.root.path, model: missing.model, s: missing)
                XCTFail("A missing prepared model must fail before stopping the current process")
            } catch EngineError.missing(let path) {
                XCTAssertTrue(path.contains("fixture/missing"))
            }
            try FileManager.default.removeItem(at: fixture.root.appendingPathComponent("build/splash"))
            do {
                try await engine.restart(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
                XCTFail("A missing binary must fail before stopping the current process")
            } catch EngineError.missing(let path) {
                XCTAssertTrue(path.hasSuffix("build/splash"))
            }
            XCTAssertEqual(engine.state, .running)
            XCTAssertEqual(engine.ownership, .owned)
            XCTAssertEqual(engine.snapshot?.pid, pid)
            XCTAssertEqual(engine.appliedSettings?.model, "fixture/base")
            XCTAssertTrue(isAlive(pid))
            let status = try await StatusClient().status(port: fixture.port)
            XCTAssertEqual(status.pid, pid)
        }
    }

    @MainActor
    func testConfiguredPortAdoptionKeepsPollingAndDisconnectsWithoutStoppingServer() async throws {
        try await withFixture { fixture, engine in
            let external = try fixture.launchExternal()
            try await fixture.waitForServer()
            await engine.adoptIfRunning(port: fixture.port)
            XCTAssertEqual(engine.port, fixture.port)
            XCTAssertEqual(engine.ownership, .external)
            let count = engine.snapshot?.submitted ?? 0
            let updated = engine.lastUpdated
            try await waitUntil { (engine.snapshot?.submitted ?? 0) > count && engine.lastUpdated != updated }
            await engine.stop()
            XCTAssertEqual(engine.state, .stopped)
            XCTAssertEqual(engine.ownership, .none)
            XCTAssertTrue(external.isRunning)
            let status = try await StatusClient().status(port: fixture.port)
            XCTAssertEqual(status.pid, external.processIdentifier)
        }
    }

    @MainActor
    func testStoppingOwnedInstanceDoesNotStopAnotherServer() async throws {
        let other = try Fixture()
        let second = try other.launchExternal()
        do {
            try await other.waitForServer()
            try await withFixture { fixture, engine in
                try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
                try await waitUntil { engine.state == .running }
                await engine.stop()
                XCTAssertTrue(second.isRunning)
                let status = try await StatusClient().status(port: other.port)
                XCTAssertEqual(status.pid, second.processIdentifier)
            }
            await other.cleanup()
        } catch {
            await other.cleanup()
            throw error
        }
    }

    @MainActor
    func testExplicitCLIStopVerifiesPIDAndPortAndPreservesAnotherServer() async throws {
        let first = try Fixture()
        let second = try Fixture()
        do {
            let target = try first.launchExternal()
            let neighbor = try second.launchExternal()
            try await first.waitForServer()
            try await second.waitForServer()

            do {
                try await ProcTree.terminateServer(pid: neighbor.processIdentifier, port: first.port)
                XCTFail("A PID that does not match the endpoint must be refused")
            } catch ProcTree.StopError.unverified {}
            do {
                try await ProcTree.terminateServer(pid: target.processIdentifier, port: second.port)
                XCTFail("A different endpoint must not authorize stopping the target")
            } catch ProcTree.StopError.unverified {}
            XCTAssertTrue(target.isRunning)
            XCTAssertTrue(neighbor.isRunning)

            try await ProcTree.terminateServer(pid: target.processIdentifier, port: first.port)
            try await waitUntil { !target.isRunning }
            XCTAssertTrue(neighbor.isRunning)
            let remaining = try await StatusClient().status(port: second.port)
            XCTAssertEqual(remaining.pid, neighbor.processIdentifier)
            await first.cleanup()
            await second.cleanup()
        } catch {
            let diagnostics = first.diagnosticLog() + "\n" + second.diagnosticLog()
            await first.cleanup()
            await second.cleanup()
            throw NSError(domain: "StudioFixture", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "CLI stop: \(error.localizedDescription)\n\(diagnostics)"])
        }
    }

    @MainActor
    func testDelayedResponseCannotOverwriteDisconnectedState() async throws {
        try await withFixture(control: ["status_delay": 0.8]) { fixture, engine in
            _ = try fixture.launchExternal()
            try await fixture.waitForServer()
            await engine.adoptIfRunning(port: fixture.port)
            let initial = fixture.integer(in: "status-requests.json", key: "count") ?? 0
            try await waitUntil { (fixture.integer(in: "status-requests.json", key: "count") ?? 0) > initial }
            await engine.stop()
            try await Task.sleep(nanoseconds: 1_100_000_000)
            XCTAssertEqual(engine.state, .stopped)
            XCTAssertEqual(engine.ownership, .none)
            XCTAssertNil(engine.snapshot)
            XCTAssertNil(engine.lastUpdated)
        }
    }

    @MainActor
    func testMemoryReleaseUsesCorrectRouteAndSurfacesRefusalWithIdleReleaseOff() async throws {
        try await withFixture { fixture, engine in
            XCTAssertEqual(fixture.settings.idleRelease, "off")
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .running }
            try await engine.releaseMemory(fixture.port)
            XCTAssertEqual(fixture.string(in: "release-requests.json", key: "path"), "/v1/memory/release")
            try fixture.writeControl(["release_status": 409])
            do {
                try await engine.releaseMemory(fixture.port)
                XCTFail("A refused memory release must throw")
            } catch StatusClient.ClientError.http(let code, _) {
                XCTAssertEqual(code, 409)
            }
            XCTAssertTrue(engine.errorMessage?.contains("409") == true)
        }
    }

    @MainActor
    func testUnavailableAndInvalidPortsDoNotCreateAnOwnedInstance() async throws {
        let port = try Fixture.availablePort()
        let engine = Engine()
        await engine.adoptIfRunning(port: port)
        XCTAssertEqual(engine.state, .stopped)
        XCTAssertEqual(engine.ownership, .none)
        XCTAssertNil(engine.snapshot)
        await engine.adoptIfRunning(port: -1)
        XCTAssertEqual(engine.state, .failed)
        XCTAssertNotNil(engine.errorMessage)
        XCTAssertEqual(engine.port, port)
        XCTAssertNotNil(URL(string: engine.endpoint.absoluteString))
        await engine.stop()
    }

    @MainActor
    func testReadinessFailureAfterStartupIsNotShownAsLoading() async throws {
        try await withFixture { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .running }
            let updated = engine.lastUpdated
            try fixture.writeControl(["ready": false])
            try await waitUntil { engine.state == .failed && engine.snapshot?.ready == false }
            XCTAssertEqual(engine.ownership, .owned)
            XCTAssertNotNil(engine.errorMessage)
            XCTAssertNotNil(engine.lastUpdated)
            XCTAssertNotEqual(engine.lastUpdated, updated)
            try fixture.writeControl(["ready": true])
            try await waitUntil { engine.state == .running }
            XCTAssertNil(engine.errorMessage)
        }
    }

    @MainActor
    func testInitialReportedMemoryAndMetalFailuresPreserveFreshDiagnostics() async throws {
        try await withFixture(control: ["ready": false, "memory_pressure": "critical"]) { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .failed && engine.snapshot?.pressure == "critical" }
            XCTAssertEqual(engine.ownership, .owned)
            XCTAssertNotNil(engine.lastUpdated)
            XCTAssertEqual(engine.errorMessage, engine.snapshot?.unavailableReason)
            try fixture.writeControl(["ready": false, "metal_healthy": false, "metal_failure_reason": "fixture GPU failure"])
            try await waitUntil { engine.errorMessage?.contains("fixture GPU failure") == true }
            XCTAssertEqual(engine.state, .failed)
            XCTAssertEqual(engine.ownership, .owned)
            XCTAssertEqual(engine.snapshot?.metalHealthy, false)
            XCTAssertNotNil(engine.lastUpdated)
        }
    }

    @MainActor
    func testInitialHealthyWarmupStaysStartingUntilReady() async throws {
        try await withFixture(control: ["ready": false]) { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.snapshot != nil }
            XCTAssertEqual(engine.state, .starting)
            XCTAssertEqual(engine.ownership, .owned)
            XCTAssertNil(engine.errorMessage)
            XCTAssertNotNil(engine.lastUpdated)
            try fixture.writeControl(["ready": true])
            try await waitUntil { engine.state == .running }
        }
    }

    @MainActor
    func testLeaderExitCleansItsOrphanChildAndAllowsRetry() async throws {
        try await withFixture(control: ["spawn_child": true]) { fixture, engine in
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .running }
            let previousInstance = engine.snapshot?.instanceID
            let childPID = try XCTUnwrap(fixture.integer(in: "child.json", key: "pid")).asPID
            XCTAssertTrue(isAlive(childPID))
            try fixture.writeControl(["exit_now": true])
            try await waitUntil(timeout: 12) { engine.state == .failed && engine.ownership == .none }
            XCTAssertFalse(isAlive(childPID))
            XCTAssertNotNil(engine.errorMessage)
            try fixture.writeControl([:])
            try await engine.start(root: fixture.root.path, model: "fixture/base", s: fixture.settings)
            try await waitUntil { engine.state == .running }
            XCTAssertNotEqual(engine.snapshot?.instanceID, previousInstance)
        }
    }
}

private enum FixtureFailure: Error { case pythonMissing, noPort, timedOut }

@MainActor
private func waitUntil(timeout: TimeInterval = 6, _ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if condition() { return }
        try await Task.sleep(nanoseconds: 50_000_000)
    }
    throw FixtureFailure.timedOut
}

@MainActor
private func withFixture(control: [String: Any] = [:], _ operation: (Fixture, Engine) async throws -> Void) async throws {
    let fixture = try Fixture(control: control)
    let engine = Engine(logURL: fixture.root.appendingPathComponent("engine.log"))
    do {
        try await operation(fixture, engine)
        await engine.stop()
        await fixture.cleanup()
    } catch {
        let details = "\(error) · state=\(engine.state) ownership=\(engine.ownership) error=\(engine.errorMessage ?? "none") log=\(engine.loadLine)\n\(fixture.diagnosticLog())"
        await engine.stop()
        await fixture.cleanup()
        throw NSError(domain: "StudioFixture", code: 1, userInfo: [NSLocalizedDescriptionKey: details])
    }
}

@MainActor
private final class Fixture {
    let root: URL
    let port: Int
    let python: String
    private var externalProcesses: [Process] = []

    var settings: Settings {
        var settings = Settings.default
        settings.splashRoot = root.path
        settings.model = "fixture/base"
        settings.port = port
        settings.context = "32K"
        settings.idleRelease = "off"
        settings.prefillInt8 = false
        settings.useANE = false
        settings.cacheDisk = "0"
        settings.persistentCache = false
        settings.maxMemory = "32G"
        return settings
    }

    init(control: [String: Any] = [:]) throws {
        let manager = FileManager.default
        // Apple's /usr/bin/python3 is a name-sensitive Xcode shim; renaming it
        // to a venv's "python" does not launch the same interpreter.
        let candidates = [ProcessInfo.processInfo.environment["SPLASH_STUDIO_TEST_PYTHON"],
                          "/opt/homebrew/bin/python3", "/usr/local/bin/python3", Self.xcodePython()].compactMap { $0 }
        guard let candidate = candidates.first(where: { manager.isExecutableFile(atPath: $0) })
        else { throw FixtureFailure.pythonMissing }
        let python = URL(fileURLWithPath: candidate).resolvingSymlinksInPath().path
        self.python = python
        port = try Self.availablePort()
        root = manager.temporaryDirectory.appendingPathComponent("splash-studio-fixture-\(UUID().uuidString)", isDirectory: true)
        for directory in [".venv/bin", "build", "server", "install/models/fixture/base/tokenizer"] {
            try manager.createDirectory(at: root.appendingPathComponent(directory), withIntermediateDirectories: true)
        }
        try manager.createSymbolicLink(atPath: root.appendingPathComponent(".venv/bin/python").path, withDestinationPath: python)
        let binary = root.appendingPathComponent("build/splash")
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: binary)
        try manager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
        try Data("{\"model\":\"fixture/base\",\"family\":\"fixture\",\"files\":{}}".utf8)
            .write(to: root.appendingPathComponent("install/models/fixture/base/model.json"))
        let resources = try XCTUnwrap(Bundle.module.resourceURL?.appendingPathComponent("Fixtures"))
        for name in ["server.py", "__init__.py"] {
            try manager.copyItem(at: resources.appendingPathComponent(name), to: root.appendingPathComponent("server/" + name))
        }
        try writeControl(control)
    }

    func writeControl(_ values: [String: Any]) throws {
        try JSONSerialization.data(withJSONObject: values).write(to: root.appendingPathComponent("fixture.json"), options: .atomic)
    }

    func launchExternal() throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.currentDirectoryURL = root
        process.arguments = ["-c", "import os,sys; os.setsid() if os.getpgrp()!=os.getpid() else None; os.execv(sys.executable,[sys.executable]+sys.argv[1:])", "-m", "server.server",
                             root.appendingPathComponent("install/models/fixture/base").path,
                             "--model", "fixture/base", "--port", String(port)]
        let logURL = root.appendingPathComponent("external.log")
        try Data().write(to: logURL)
        let log = try FileHandle(forWritingTo: logURL)
        process.standardOutput = log
        process.standardError = log
        try process.run()
        try? log.close()
        externalProcesses.append(process)
        return process
    }

    func waitForServer() async throws {
        let deadline = Date().addingTimeInterval(6)
        while Date() < deadline {
            if let status = try? await StatusClient().status(port: port), status.ready { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw NSError(domain: "StudioFixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "Fixture server did not become ready.\n" + diagnosticLog()])
    }

    func integer(in name: String, key: String) -> Int? { object(name)?[key] as? Int }
    func string(in name: String, key: String) -> String? { object(name)?[key] as? String }

    private func object(_ name: String) -> [String: Any]? {
        guard let data = try? Data(contentsOf: root.appendingPathComponent(name)) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    func diagnosticLog() -> String {
        ["engine.log", "external.log"].compactMap { name in
            guard let data = try? Data(contentsOf: root.appendingPathComponent(name)) else { return nil }
            return name + ":\n" + String(decoding: data.suffix(8192), as: UTF8.self)
        }.joined(separator: "\n")
    }

    func cleanup() async {
        for process in externalProcesses {
            _ = await ProcTree.terminateOwned(process)
        }
        if let child = integer(in: "child.json", key: "pid"), child > 1,
           let group = integer(in: "child.json", key: "group"), group > 1 {
            let rootMarker = root.path
            await Task.detached {
                // A fallback for a failed lifecycle assertion may only signal
                // the identified fixture child, never a recycled bare PID.
                let query = Process()
                query.executableURL = URL(fileURLWithPath: "/bin/ps")
                query.arguments = ["-p", String(child), "-o", "command="]
                let pipe = Pipe()
                query.standardOutput = pipe
                query.standardError = FileHandle.nullDevice
                if (try? query.run()) != nil {
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    query.waitUntilExit()
                    let command = String(decoding: data, as: UTF8.self)
                    if command.contains("splash-studio-fixture-child"), command.contains(rootMarker),
                       getpgid(Int32(child)) == Int32(group), Int32(group) != getpgrp() {
                        kill(Int32(child), SIGKILL)
                    }
                }
            }.value
        }
        try? FileManager.default.removeItem(at: root)
    }

    static func availablePort() throws -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw FixtureFailure.noPort }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { throw FixtureFailure.noPort }
        var size = socklen_t(MemoryLayout<sockaddr_in>.size)
        let result = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &size) }
        }
        guard result == 0 else { throw FixtureFailure.noPort }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    private static func xcodePython() -> String? {
        let locator = Process()
        locator.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        locator.arguments = ["--find", "python3"]
        let pipe = Pipe()
        locator.standardOutput = pipe
        locator.standardError = FileHandle.nullDevice
        guard (try? locator.run()) != nil else { return nil }
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        locator.waitUntilExit()
        guard locator.terminationStatus == 0 else { return nil }
        return String(decoding: output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private func isAlive(_ pid: Int32) -> Bool { kill(pid, 0) == 0 || errno == EPERM }
private extension Int { var asPID: Int32 { Int32(self) } }
