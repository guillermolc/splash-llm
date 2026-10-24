import Foundation

struct StatusClient {
    enum ClientError: LocalizedError {
        case invalidPort, invalidResponse, http(Int, String)
        var errorDescription: String? {
            switch self {
            case .invalidPort: return "El puerto debe estar entre 1 y 65535."
            case .invalidResponse: return "El servidor devolvió una respuesta no válida."
            case .http(let code, let message): return "HTTP \(code): \(message)"
            }
        }
    }

    private let session: URLSession
    private let apiKey: String?
    init(session: URLSession = .shared, apiKey: String? = ProcessInfo.processInfo.environment["SPLASH_API_KEY"]) {
        self.session = session
        self.apiKey = apiKey
    }

    func status(port: Int) async throws -> Snapshot {
        let data = try await request(port: port, path: "/status")
        return try StatusParser.parse(data: data)
    }

    func releaseMemory(port: Int) async throws {
        let data = try await request(port: port, path: "/v1/memory/release", method: "POST")
        guard let response = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              response["released"] as? Bool == true else { throw ClientError.invalidResponse }
    }

    private func request(port: Int, path: String, method: String = "GET") async throws -> Data {
        guard (1...65535).contains(port), let url = URL(string: "http://127.0.0.1:\(port)\(path)")
        else { throw ClientError.invalidPort }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = method == "GET" ? 2 : 8
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if let apiKey, !apiKey.isEmpty { request.setValue("Bearer " + apiKey, forHTTPHeaderField: "Authorization") }
        if method == "POST" {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("{}".utf8)
        }
        let (data, response) = try await session.data(for: request)
        try Task.checkCancellation()
        guard let response = response as? HTTPURLResponse, data.count <= 2 * 1024 * 1024
        else { throw ClientError.invalidResponse }
        guard (200..<300).contains(response.statusCode) else {
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let detail = (json?["error"] as? [String: Any])?["message"] as? String
            throw ClientError.http(response.statusCode, detail ?? HTTPURLResponse.localizedString(forStatusCode: response.statusCode))
        }
        return data
    }
}
