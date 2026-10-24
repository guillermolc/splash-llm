import Foundation
import XCTest
@testable import splash_studio

final class ConfigStatusTests: XCTestCase {
    private var temporaryDirectories: [URL] = []

    override func tearDownWithError() throws {
        for directory in temporaryDirectories {
            try FileManager.default.removeItem(at: directory)
        }
        temporaryDirectories = []
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("splash-studio-settings-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        temporaryDirectories.append(directory)
        return directory
    }

    private func validSettings(root: URL) -> Settings {
        var settings = Settings.default
        settings.splashRoot = root.path
        settings.model = "test/model"
        return settings
    }

    private func status(_ object: [String: Any]) throws -> Snapshot {
        try StatusParser.parse(data: JSONSerialization.data(withJSONObject: object))
    }

    func testSettingsPersistAtomicallyAndReload() throws {
        let root = try temporaryDirectory()
        let url = root.appendingPathComponent("preferences/config.json")
        let defaults = validSettings(root: root)
        let config = Config(fileURL: url, defaults: defaults)
        var updated = defaults
        updated.port = 8123
        updated.context = "64K"
        updated.persistentCache = false
        try config.save(updated)

        XCTAssertEqual(config.settings, updated)
        XCTAssertEqual(Config(fileURL: url, defaults: defaults).settings, updated)
        XCTAssertEqual(try JSONDecoder().decode(Settings.self, from: Data(contentsOf: url)), updated)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path),
                       ["config.json"])
    }

    func testMissingFieldsMigrateIndividuallyUsingInjectedDefaults() throws {
        let root = try temporaryDirectory()
        let url = root.appendingPathComponent("config.json")
        var defaults = validSettings(root: root)
        defaults.useANE = false
        defaults.cacheDisk = "8G"
        try Data(#"{"model":"test/legacy","port":8123,"context":"16K"}"#.utf8).write(to: url)

        let config = Config(fileURL: url, defaults: defaults)
        XCTAssertNil(config.loadError)
        XCTAssertEqual(config.settings.model, "test/legacy")
        XCTAssertEqual(config.settings.port, 8123)
        XCTAssertEqual(config.settings.context, "16K")
        XCTAssertEqual(config.settings.splashRoot, root.path)
        XCTAssertEqual(config.settings.kvFormat, defaults.kvFormat)
        XCTAssertEqual(config.settings.useANE, false)
        XCTAssertEqual(config.settings.cacheDisk, "8G")
        XCTAssertEqual(config.settings.persistentCache, defaults.persistentCache)
    }

    func testCorruptConfigurationIsReportedAndPreserved() throws {
        let root = try temporaryDirectory()
        let url = root.appendingPathComponent("config.json")
        let damaged = Data("{not valid JSON".utf8)
        try damaged.write(to: url)
        let defaults = validSettings(root: root)

        let config = Config(fileURL: url, defaults: defaults)
        XCTAssertNotNil(config.loadError)
        XCTAssertEqual(config.settings, defaults)
        XCTAssertEqual(try Data(contentsOf: url), damaged)
    }

    func testFailedWriteDoesNotChangeInMemorySettingsOrBlockingFile() throws {
        let root = try temporaryDirectory()
        let blocker = root.appendingPathComponent("not-a-directory")
        let original = Data("keep this file".utf8)
        try original.write(to: blocker)
        let defaults = validSettings(root: root)
        let config = Config(fileURL: blocker.appendingPathComponent("config.json"), defaults: defaults)
        var updated = defaults
        updated.port = 8123

        XCTAssertThrowsError(try config.save(updated))
        XCTAssertEqual(config.settings, defaults)
        XCTAssertEqual(try Data(contentsOf: blocker), original)
    }

    func testConfigStoreSavesChangedValuesAndFreshStoreReadsThem() async throws {
        let root = try temporaryDirectory()
        let url = root.appendingPathComponent("config.json")
        let defaults = validSettings(root: root)
        var edited = defaults
        edited.port = 8456
        edited.context = "auto"
        edited.maxMemory = "32G"
        let updated = edited

        try await MainActor.run {
            let store = ConfigStore(fileURL: url, defaults: defaults)
            try store.save(updated)
            XCTAssertEqual(store.settings, updated)
            XCTAssertNil(store.errorMessage)
            let reloaded = ConfigStore(fileURL: url, defaults: defaults)
            XCTAssertEqual(reloaded.settings, updated)
        }
    }

    func testConfigStoreRejectsInvalidChangesWithoutWriting() async throws {
        let root = try temporaryDirectory()
        let url = root.appendingPathComponent("config.json")
        let defaults = validSettings(root: root)
        var edited = defaults
        edited.port = 70_000
        let invalid = edited

        try await MainActor.run {
            let store = ConfigStore(fileURL: url, defaults: defaults)
            XCTAssertThrowsError(try store.save(invalid))
            XCTAssertEqual(store.settings, defaults)
            XCTAssertNotNil(store.errorMessage)
            store.dismissError()
            XCTAssertNil(store.errorMessage)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testPortValidationRejectsOutsideRange() throws {
        let root = try temporaryDirectory()
        var settings = validSettings(root: root)
        for invalid in [0, -1, 65_536] {
            settings.port = invalid
            XCTAssertTrue(settings.validationErrors().contains { $0.contains("puerto") })
        }
        for valid in [1, 8001, 65_535] {
            settings.port = valid
            XCTAssertTrue(settings.validationErrors().isEmpty)
        }
    }

    func testContextValidationMirrorsServerAndModelLimits() throws {
        let root = try temporaryDirectory()
        var settings = validSettings(root: root)
        for invalid in ["0", "-1", "bogus", "512K", "999999999999999999999K"] {
            settings.context = invalid
            XCTAssertTrue(settings.validationErrors().contains { $0.contains("contexto") })
        }
        settings.context = "64K"
        XCTAssertEqual(settings.contextTokens, 65_536)
        XCTAssertFalse(settings.validationErrors(maximumContextTokens: 32_768).isEmpty)
        settings.context = "  auto  "
        XCTAssertNil(settings.contextTokens)
        XCTAssertTrue(settings.validationErrors(maximumContextTokens: 32_768).isEmpty)
        settings.context = "256K"
        XCTAssertTrue(settings.validationErrors().isEmpty)
    }

    func testMemoryCacheAndIdleValidation() throws {
        let root = try temporaryDirectory()
        var settings = validSettings(root: root)
        for value in ["auto", "32G", "32GiB", "8192M", "4096"] {
            settings.maxMemory = value
            XCTAssertTrue(settings.validationErrors().isEmpty, value)
        }
        for value in ["0", "-32G", "32.5G", "infinity", "999999999999999999999G"] {
            settings.maxMemory = value
            XCTAssertFalse(settings.validationErrors().isEmpty, value)
        }
        settings.maxMemory = "auto"
        settings.cacheDisk = "0"
        XCTAssertTrue(settings.validationErrors().isEmpty)
        settings.cacheDisk = "-1G"
        XCTAssertFalse(settings.validationErrors().isEmpty)
        settings.cacheDisk = "auto"
        for value in ["off", "30m", "1.5h", "600", "0.5s"] {
            settings.idleRelease = value
            XCTAssertTrue(settings.validationErrors().isEmpty, value)
        }
        for value in ["0", "-1s", "nan", "inf", "3d"] {
            settings.idleRelease = value
            XCTAssertFalse(settings.validationErrors().isEmpty, value)
        }
    }

    func testRootAndModelValidationPreservesOriginalInput() throws {
        let root = try temporaryDirectory()
        var settings = validSettings(root: root)
        settings.splashRoot = " \(root.path) "
        XCTAssertTrue(settings.validationErrors().isEmpty)
        XCTAssertEqual(settings.splashRoot, " \(root.path) ")
        settings.splashRoot = root.appendingPathComponent("missing").path
        XCTAssertFalse(settings.validationErrors().isEmpty)
        settings.splashRoot = root.path
        for model in ["", "../model", "owner/../model", "owner/model --flag", "owner/model.git"] {
            settings.model = model
            XCTAssertFalse(settings.validationErrors().isEmpty, model)
        }
        settings.model = "test/model:UD-Q4_K_M"
        XCTAssertTrue(settings.validationErrors().isEmpty)
    }

    func testGoldenStatusDecodesWithTruthfulUnitsAndAvailability() throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repository.appendingPathComponent("dev/tests/engine/status_golden.json"))
        let value = try StatusParser.parse(data: data)

        XCTAssertEqual(value.schemaVersion, 6)
        XCTAssertTrue(value.schemaSupported)
        XCTAssertTrue(value.isValid)
        XCTAssertTrue(value.ready)
        XCTAssertTrue(value.hasRequestMetrics)
        XCTAssertTrue(value.hasSchedulerMetrics)
        XCTAssertTrue(value.hasActiveRequests)
        XCTAssertEqual(value.submitted, 3)
        XCTAssertEqual(value.inFlight.count, 2)
        XCTAssertEqual(value.kvResidentTokens, (128 - 72) * 32)
        XCTAssertEqual(value.poolTokens, 128 * 32)
        XCTAssertEqual(try XCTUnwrap(value.cacheHit), 100.0 / 3.0, accuracy: 0.000001)
        XCTAssertEqual(value.ramGB, 32)
        XCTAssertEqual(value.diskPersistent, true)
        XCTAssertEqual(value.aneState, "stopped")
        XCTAssertNil(value.itlMs)
        XCTAssertNil(value.ttftMs)
    }

    func testPartialStatusDoesNotInventMeasurements() throws {
        let value = try status(["schema_version": 6, "ready": false])
        XCTAssertTrue(value.isValid)
        XCTAssertFalse(value.ready)
        XCTAssertFalse(value.hasRequestMetrics)
        XCTAssertFalse(value.hasSchedulerMetrics)
        XCTAssertFalse(value.hasActiveRequests)
        XCTAssertNil(value.memAlloc)
        XCTAssertNil(value.pressure)
        XCTAssertNil(value.weightsState)
        XCTAssertNil(value.kvResidentTokens)
        XCTAssertNil(value.cacheHit)
    }

    func testMeasuredZeroRemainsDistinctFromUnavailable() throws {
        let value = try status([
            "schema_version": 6, "ready": true,
            "requests": ["submitted": 0, "completed": 0, "cancelled": 0, "failed": 0],
            "scheduler": ["queued": 0, "prefilling": 0, "decoding": 0],
            "active_requests": [],
            "metrics": ["ttft_ms": ["p50": 0, "samples": 1],
                        "itl_ms": ["p50": 0, "samples": 0]]
        ])
        XCTAssertTrue(value.hasRequestMetrics)
        XCTAssertTrue(value.hasSchedulerMetrics)
        XCTAssertTrue(value.hasActiveRequests)
        XCTAssertEqual(value.submitted, 0)
        XCTAssertTrue(value.inFlight.isEmpty)
        XCTAssertEqual(value.ttftMs, 0)
        XCTAssertNil(value.itlMs)
    }

    func testUnknownMalformedAndMissingSchemaAreRejected() throws {
        let documents: [[String: Any]] = [
            ["schema_version": 7, "ready": true],
            ["ready": true],
            ["schema_version": 6],
            ["schema_version": 6, "ready": "true"]
        ]
        for invalid in documents {
            XCTAssertThrowsError(try status(invalid))
            XCTAssertFalse(StatusParser.parse(invalid).isValid)
        }
        XCTAssertThrowsError(try StatusParser.parse(data: Data("[]".utf8)))
    }

    func testOverlappingKVPageOwnersAreNotAddedTogether() throws {
        let value = try status([
            "schema_version": 6, "ready": true,
            "kv": ["pages_allocated": 100, "pages_active": 60, "pages_cache": 70,
                   "pages_free": 10, "block_tokens": 32]
        ])
        XCTAssertEqual(value.kvActivePages, 60)
        XCTAssertEqual(value.kvCachePages, 70)
        XCTAssertEqual(value.kvResidentTokens, 90 * 32)
        XCTAssertEqual(value.ctxUsed, 90 * 32)
        let incomplete = try status([
            "schema_version": 6, "ready": true,
            "kv": ["pages_allocated": 100, "pages_active": 60, "pages_cache": 70, "block_tokens": 32]
        ])
        XCTAssertNil(incomplete.kvResidentTokens)
    }

    func testStatusUsesCurrentDeviceMemoryAndValidIdentity() throws {
        let value = try status([
            "schema_version": 6, "ready": true,
            "instance": ["id": "fixture-instance", "pid": 1234, "model": "test/model", "port": 8123],
            "memory_actual": ["allocated_bytes": 1024, "current_bytes": 2048],
            "weights": ["released": false, "restoring": false, "idle_release_seconds": NSNull()]
        ])
        XCTAssertEqual(value.memAlloc, 2048)
        XCTAssertEqual(value.instanceID, "fixture-instance")
        XCTAssertEqual(value.pid, 1234)
        XCTAssertEqual(value.port, 8123)
        XCTAssertEqual(value.weightsState, "cargados")
        XCTAssertNil(value.weightsIdleMin)
    }

    func testCriticalMemoryPressureExplainsUnavailableReadiness() throws {
        let value = try status([
            "schema_version": 6, "ready": false, "memory_pressure": "critical",
            "metal": ["healthy": true, "failure_reason": ""]
        ])
        XCTAssertEqual(value.metalHealthy, true)
        XCTAssertTrue(try XCTUnwrap(value.unavailableReason).contains("presión crítica de memoria"))
    }

    func testMetalFailureExplainsUnavailableReadinessUsingReportedReason() throws {
        let value = try status([
            "schema_version": 6, "ready": false,
            "metal": ["healthy": false, "failure_reason": "fixture command failed"]
        ])
        XCTAssertEqual(value.metalHealthy, false)
        XCTAssertEqual(value.metalFailureReason, "fixture command failed")
        XCTAssertTrue(try XCTUnwrap(value.unavailableReason).contains("fixture command failed"))
        let noDetail = try status([
            "schema_version": 6, "ready": false,
            "metal": ["healthy": false, "failure_reason": " "]
        ])
        XCTAssertEqual(noDetail.unavailableReason, "El motor informó un error de GPU.")
    }

    func testReadinessWithoutHealthEvidenceHasNoInventedFailureReason() throws {
        let unknown = try status([
            "schema_version": 6, "ready": false, "memory_pressure": "normal",
            "metal": ["healthy": true, "failure_reason": ""]
        ])
        XCTAssertNil(unknown.unavailableReason)
        let ready = try status([
            "schema_version": 6, "ready": true, "memory_pressure": "critical",
            "metal": ["healthy": false, "failure_reason": "ignored when ready"]
        ])
        XCTAssertNil(ready.unavailableReason)
    }

    func testMalformedActiveRequestDoesNotBecomeAnEmptyMeasuredList() throws {
        let value = try status([
            "schema_version": 6, "ready": true,
            "active_requests": [["id": 3, "phase": "decode"]]
        ])
        XCTAssertFalse(value.hasActiveRequests)
        XCTAssertTrue(value.inFlight.isEmpty)
    }

    func testLogRowsUseProvidedIdentityAndRejectInvalidNumbers() throws {
        let line = "12:00:01 Done · input 1,024 · cached 512 · output 128 · TTFT 0.5s · 42.0 tok/s"
        let first = try XCTUnwrap(LogParser.done(line, id: "session:1"))
        let second = try XCTUnwrap(LogParser.done(line, id: "session:2"))
        XCTAssertNotEqual(first.id, second.id)
        XCTAssertEqual(first.input, 1024)
        XCTAssertEqual(first.cached, 512)
        XCTAssertEqual(first.output, 128)
        XCTAssertEqual(first.ttft, 0.5)
        XCTAssertEqual(first.tps, 42)
        XCTAssertNil(LogParser.done(line.replacingOccurrences(of: "42.0", with: "nan")))
        XCTAssertNil(LogParser.done(line.replacingOccurrences(of: "128", with: "-128")))
    }

    func testBinaryMemoryFormattingAndInvalidMeasurements() {
        XCTAssertEqual(Fmt.bytes(1_073_741_824), "1.0 GiB")
        XCTAssertEqual(Fmt.bytes(1_048_576), "1 MiB")
        XCTAssertEqual(Fmt.bytes(1024), "1 KiB")
        XCTAssertEqual(Fmt.bytes(0), "0 B")
        XCTAssertEqual(Fmt.bytes(.nan), "—")
        XCTAssertEqual(Fmt.ms(-1), "—")
        XCTAssertEqual(Fmt.tokD(.infinity), "—")
    }
}
