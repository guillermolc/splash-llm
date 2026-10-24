import Combine
import Foundation

private extension CodingUserInfoKey {
    static let studioSettingsDefaults = CodingUserInfoKey(rawValue: "StudioSettingsDefaults")!
}

struct Settings: Codable, Equatable, Sendable {
    var splashRoot: String
    var model: String
    var port: Int
    var context: String
    var kvFormat: String
    var idleRelease: String
    var prefillInt8: Bool
    var maxMemory: String
    var useANE: Bool
    var cacheDisk: String
    var persistentCache: Bool

    static let maximumSupportedContext = 262_144

    static let `default` = Settings(
        splashRoot: NSHomeDirectory() + "/Documents/Projects/splash-llm",
        model: "mlx-community/Qwen3.8-27B-4bit",
        port: 8001,
        context: "32K",
        kvFormat: "int8",
        idleRelease: "30m",
        prefillInt8: true,
        maxMemory: "auto",
        useANE: true,
        cacheDisk: "auto",
        persistentCache: true
    )

    /// Missing fields inherit defaults individually, preserving older settings.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = decoder.userInfo[.studioSettingsDefaults] as? Settings ?? Self.default
        splashRoot = try c.decodeIfPresent(String.self, forKey: .splashRoot) ?? defaults.splashRoot
        model = try c.decodeIfPresent(String.self, forKey: .model) ?? defaults.model
        port = try c.decodeIfPresent(Int.self, forKey: .port) ?? defaults.port
        context = try c.decodeIfPresent(String.self, forKey: .context) ?? defaults.context
        kvFormat = try c.decodeIfPresent(String.self, forKey: .kvFormat) ?? defaults.kvFormat
        idleRelease = try c.decodeIfPresent(String.self, forKey: .idleRelease) ?? defaults.idleRelease
        prefillInt8 = try c.decodeIfPresent(Bool.self, forKey: .prefillInt8) ?? defaults.prefillInt8
        maxMemory = try c.decodeIfPresent(String.self, forKey: .maxMemory) ?? defaults.maxMemory
        useANE = try c.decodeIfPresent(Bool.self, forKey: .useANE) ?? defaults.useANE
        cacheDisk = try c.decodeIfPresent(String.self, forKey: .cacheDisk) ?? defaults.cacheDisk
        persistentCache = try c.decodeIfPresent(Bool.self, forKey: .persistentCache) ?? defaults.persistentCache
    }

    init(splashRoot: String, model: String, port: Int, context: String,
         kvFormat: String, idleRelease: String, prefillInt8: Bool,
         maxMemory: String, useANE: Bool, cacheDisk: String,
         persistentCache: Bool = true) {
        self.splashRoot = splashRoot
        self.model = model
        self.port = port
        self.context = context
        self.kvFormat = kvFormat
        self.idleRelease = idleRelease
        self.prefillInt8 = prefillInt8
        self.maxMemory = maxMemory
        self.useANE = useANE
        self.cacheDisk = cacheDisk
        self.persistentCache = persistentCache
    }

    var contextTokens: Int? {
        let text = context.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard text != "AUTO" else { return nil }
        let multiplier = text.hasSuffix("K") ? 1024 : 1
        let digits = text.hasSuffix("K") ? String(text.dropLast()) : text
        guard let count = Int(digits), count > 0 else { return nil }
        let (tokens, overflow) = count.multipliedReportingOverflow(by: multiplier)
        return overflow ? nil : tokens
    }

    /// Mirror the server's option syntax; validation never rewrites user input.
    func validationErrors(maximumContextTokens: Int? = nil) -> [String] {
        var errors: [String] = []
        if !(1...65535).contains(port) {
            errors.append("El puerto debe estar entre 1 y 65535.")
        }
        let root = splashRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        if root.isEmpty {
            errors.append("Selecciona la carpeta de Splash.")
        } else {
            var isDirectory: ObjCBool = false
            let expanded = (root as NSString).expandingTildeInPath
            if !FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory)
                || !isDirectory.boolValue {
                errors.append("La carpeta de Splash no existe.")
            }
        }
        let modelPattern = #"^[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$"#
        let modelParts = model.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        let repository = String(modelParts.first ?? "")
        let variant = modelParts.count == 2 ? String(modelParts[1]) : nil
        if model.range(of: modelPattern, options: .regularExpression) == nil
            || repository.contains("..") || repository.contains("--") || repository.hasSuffix(".git")
            || variant?.contains("..") == true {
            errors.append("Selecciona un modelo válido con formato organización/modelo.")
        }
        let contextText = context.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if contextText != "auto" {
            let limit = min(maximumContextTokens ?? Self.maximumSupportedContext, Self.maximumSupportedContext)
            if let tokens = contextTokens {
                if tokens > limit {
                    errors.append("El contexto supera el máximo compatible: \(limit) tokens.")
                }
            } else {
                errors.append("El contexto debe ser auto o una cantidad positiva, como 32K.")
            }
        }
        if kvFormat != "int8" && kvFormat != "bf16" {
            errors.append("El formato KV debe ser int8 o bf16.")
        }
        if !Self.validByteCount(maxMemory, allowZero: false) {
            errors.append("La memoria debe ser auto o una cantidad positiva, como 32G.")
        }
        if !Self.validByteCount(cacheDisk, allowZero: true) {
            errors.append("La caché SSD debe ser auto, 0 o una cantidad positiva, como 16G.")
        }
        if !Self.validDuration(idleRelease) {
            errors.append("La liberación por inactividad debe ser off o un tiempo positivo, como 30m.")
        }
        return errors
    }

    private static func validByteCount(_ value: String, allowZero: Bool) -> Bool {
        let text = value.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        if text == "AUTO" || (allowZero && text == "0") { return true }
        var digits = text
        var multiplier: Int64 = 1
        let units: [(String, Int64)] = [
            ("GIB", 1 << 30), ("GB", 1 << 30), ("G", 1 << 30),
            ("MIB", 1 << 20), ("MB", 1 << 20), ("M", 1 << 20),
            ("KIB", 1 << 10), ("KB", 1 << 10), ("K", 1 << 10)
        ]
        for (suffix, amount) in units where text.hasSuffix(suffix) {
            digits = String(text.dropLast(suffix.count))
            multiplier = amount
            break
        }
        guard !digits.isEmpty, digits.allSatisfy({ $0.isASCII && $0.isNumber }),
              let count = Int64(digits), count > 0 else { return false }
        return !count.multipliedReportingOverflow(by: multiplier).overflow
    }

    private static func validDuration(_ value: String) -> Bool {
        let text = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if text == "off" { return true }
        guard !text.isEmpty else { return false }
        let multiplier: Double
        let digits: String
        if let unit = text.last, "smh".contains(unit) {
            multiplier = unit == "h" ? 3600 : unit == "m" ? 60 : 1
            digits = String(text.dropLast())
        } else {
            multiplier = 1
            digits = text
        }
        guard let count = Double(digits) else { return false }
        let seconds = count * multiplier
        return seconds.isFinite && seconds > 0
    }
}

enum ConfigurationError: LocalizedError {
    case invalid([String])
    case load(String)
    case save(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let errors): return errors.joined(separator: "\n")
        case .load(let detail): return "No se pudo leer la configuración guardada. \(detail)"
        case .save(let detail): return "No se pudo guardar la configuración. \(detail)"
        }
    }
}

final class Config {
    let fileURL: URL
    var settings: Settings
    private(set) var loadError: String?
    private var pathError: String?

    static var directoryURL: URL {
        if let value = ProcessInfo.processInfo.environment["SPLASH_STUDIO_CONFIG"] {
            let path = (value as NSString).expandingTildeInPath
            guard !path.isEmpty, path.hasPrefix("/") else {
                return FileManager.default.temporaryDirectory
                    .appendingPathComponent("splash-studio-invalid-profile", isDirectory: true)
            }
            return URL(fileURLWithPath: path)
                .standardizedFileURL.deletingLastPathComponent()
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "Library/Application Support/splash-menubar", isDirectory: true)
    }

    static var defaultFileURL: URL {
        if let value = ProcessInfo.processInfo.environment["SPLASH_STUDIO_CONFIG"] {
            let path = (value as NSString).expandingTildeInPath
            if !path.isEmpty, path.hasPrefix("/") {
                return URL(fileURLWithPath: path).standardizedFileURL
            }
        }
        return directoryURL.appendingPathComponent("config.json")
    }

    init(fileURL: URL = Config.defaultFileURL, defaults: Settings = .default) {
        self.fileURL = fileURL
        settings = defaults
        if fileURL == Self.defaultFileURL,
           let value = ProcessInfo.processInfo.environment["SPLASH_STUDIO_CONFIG"],
           value.isEmpty || !(value as NSString).expandingTildeInPath.hasPrefix("/") {
            let message = "SPLASH_STUDIO_CONFIG debe indicar una ruta absoluta no vacía."
            pathError = message
            loadError = message
            return
        }
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        do {
            let decoder = JSONDecoder()
            decoder.userInfo[.studioSettingsDefaults] = defaults
            settings = try decoder.decode(Settings.self, from: Data(contentsOf: fileURL))
        } catch {
            // Keep the unreadable file intact; launching the app must not erase it.
            loadError = ConfigurationError.load(error.localizedDescription).localizedDescription
        }
    }

    var engineLogURL: URL { Self.logFileURL() }

    static func logFileURL() -> URL {
        directoryURL.appendingPathComponent("logs/engine.log")
    }

    func save(_ updated: Settings) throws {
        if let pathError { throw ConfigurationError.save(pathError) }
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(updated)
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: fileURL, options: .atomic)
            settings = updated
            loadError = nil
        } catch {
            throw ConfigurationError.save(error.localizedDescription)
        }
    }

    func save() throws { try save(settings) }
}

@MainActor
final class ConfigStore: ObservableObject {
    static let shared = ConfigStore()

    @Published private(set) var settings: Settings
    @Published private(set) var errorMessage: String?
    private let config: Config

    init(fileURL: URL = Config.defaultFileURL, defaults: Settings = .default) {
        config = Config(fileURL: fileURL, defaults: defaults)
        settings = config.settings
        errorMessage = config.loadError
    }

    func save(_ updated: Settings) throws {
        let errors = updated.validationErrors()
        guard errors.isEmpty else {
            let error = ConfigurationError.invalid(errors)
            errorMessage = error.localizedDescription
            throw error
        }
        do {
            try config.save(updated)
            settings = updated
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            throw error
        }
    }

    func dismissError() { errorMessage = nil }
}
