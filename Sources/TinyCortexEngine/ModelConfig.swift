import Foundation

/// Parsed JSON manifest describing TCN weight layout.
public struct ModelConfig: Codable {
    public let architecture: String
    public let channels: Int
    public let blocks: Int
    public let kernelSize: Int
    public let receptiveField: Int
    public let totalParameters: Int
    public let totalBytes: Int
    public let layers: [LayerInfo]

    enum CodingKeys: String, CodingKey {
        case architecture, channels, blocks
        case kernelSize = "kernel_size"
        case receptiveField = "receptive_field"
        case totalParameters = "total_parameters"
        case totalBytes = "total_bytes"
        case layers
    }

    public struct LayerInfo: Codable {
        public let name: String
        public let shape: [Int]
        public let dtype: String
        public let byteOffset: Int
        public let byteLength: Int

        enum CodingKeys: String, CodingKey {
            case name, shape, dtype
            case byteOffset = "byte_offset"
            case byteLength = "byte_length"
        }
    }

    public static func load(from url: URL) throws -> ModelConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ModelConfig.self, from: data)
    }

    /// Byte offset for a named layer in the weight binary.
    public func offset(for layerName: String) -> Int {
        guard let layer = layers.first(where: { $0.name == layerName }) else {
            fatalError("Layer '\(layerName)' not found in model config")
        }
        return layer.byteOffset
    }

    /// Per-block byte stride (dilated conv weight/bias + prelu + 1x1 weight/bias).
    public var blockByteStride: Int {
        // blocks.0 starts at offset of blocks.0.net.0.conv.weight
        // blocks.1 starts at offset of blocks.1.net.0.conv.weight
        let block0Start = offset(for: "blocks.0.net.0.conv.weight")
        let block1Start = offset(for: "blocks.1.net.0.conv.weight")
        return block1Start - block0Start
    }
}
