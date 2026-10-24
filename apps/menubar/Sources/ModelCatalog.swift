import Foundation

/// Prepared manifests provide sizes without walking weight directories.
actor ModelCatalog {
    private var cache: [String: (Date, [Engine.Model])] = [:]

    func models(root: String, force: Bool) -> [Engine.Model] {
        let root = (root.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).expandingTildeInPath
        if !force, let (date, models) = cache[root], Date().timeIntervalSince(date) < 30 { return models }
        let base = URL(fileURLWithPath: root).appendingPathComponent("install/models", isDirectory: true)
        let manager = FileManager.default
        let orgs = (try? manager.contentsOfDirectory(at: base, includingPropertiesForKeys: [.isDirectoryKey], options: .skipsHiddenFiles)) ?? []
        var result: [Engine.Model] = []
        for org in orgs {
            let repos = (try? manager.contentsOfDirectory(at: org, includingPropertiesForKeys: [.isDirectoryKey], options: .skipsHiddenFiles)) ?? []
            for repo in repos {
                guard let data = try? Data(contentsOf: repo.appendingPathComponent("model.json")),
                      let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
                var unique: [String: Int64] = [:]
                for (_, value) in manifest["files"] as? [String: [String: Any]] ?? [:] {
                    if let path = value["path"] as? String, let bytes = value["bytes"] as? NSNumber { unique[path] = bytes.int64Value }
                }
                var maximum: Int?
                for path in ["target/config.json", "config.json"] {
                    guard let configData = try? Data(contentsOf: repo.appendingPathComponent(path)),
                          let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else { continue }
                    let text = config["text_config"] as? [String: Any] ?? config
                    maximum = text["max_position_embeddings"] as? Int
                    if maximum != nil { break }
                }
                result.append(Engine.Model(id: org.lastPathComponent + "/" + repo.lastPathComponent,
                                           sizeGB: Double(unique.values.reduce(0, +)) / 1_000_000_000,
                                           maximumContextTokens: maximum))
            }
        }
        result.sort { $0.id < $1.id }
        cache[root] = (Date(), result)
        return result
    }
}
