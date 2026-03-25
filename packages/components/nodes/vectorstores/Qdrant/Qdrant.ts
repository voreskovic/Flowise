import { flatten } from 'lodash'
import { v4 as uuid } from 'uuid'
import { QdrantClient } from '@qdrant/js-client-rest'
import { VectorStoreRetrieverInput } from '@langchain/core/vectorstores'
import { Document } from '@langchain/core/documents'
import { QdrantVectorStore, QdrantLibArgs } from '@langchain/qdrant'
import { Embeddings } from '@langchain/core/embeddings'
import { ICommonObject, INode, INodeData, INodeOutputsValue, INodeParams, IndexingResult } from '../../../src/Interface'
import { FLOWISE_CHATID, getBaseClasses, getCredentialData, getCredentialParam, parseJsonBody } from '../../../src/utils'
import { index } from '../../../src/indexing'
import { howToUseFileUpload } from '../VectorStoreUtils'

type RetrieverConfig = Partial<VectorStoreRetrieverInput<QdrantVectorStore>>
type QdrantAddDocumentOptions = {
    customPayload?: Record<string, any>[]
    ids?: string[]
}

/**
 * Parse the MRL dimensions string into an array of integers.
 * E.g. "512, 256, 128" -> [512, 256, 128]
 */
function parseMrlDimensions(raw: string): number[] {
    return raw
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !isNaN(n) && n > 0)
}

/**
 * Build Qdrant named-vectors config for MRL.
 * Returns e.g. { "full": { size: 1024, distance: "Cosine" }, "512": { size: 512, distance: "Cosine" } }
 * Truncated vectors use their dimension as the vector name (e.g. "512", "256").
 */
function buildMrlVectorsConfig(fullDimension: number, truncatedDimensions: number[], distance: string): Record<string, any> {
    const vectors: Record<string, any> = {
        full: { size: fullDimension, distance }
    }
    for (const dim of truncatedDimensions) {
        if (dim >= fullDimension) continue
        vectors[`${dim}`] = { size: dim, distance }
    }
    return vectors
}

/**
 * Truncate an embedding vector to the specified length and L2-normalize it.
 * MRL embeddings retain semantic meaning when truncated, but re-normalization
 * is recommended for cosine similarity to remain well-behaved.
 */
function truncateAndNormalize(embedding: number[], targetDim: number): number[] {
    const truncated = embedding.slice(0, targetDim)
    const norm = Math.sqrt(truncated.reduce((sum, v) => sum + v * v, 0))
    if (norm === 0) return truncated
    return truncated.map((v) => v / norm)
}

/**
 * Build the named-vector map for a single point during MRL upsert.
 */
function buildMrlVectorMap(embedding: number[], truncatedDimensions: number[], fullDimension: number): Record<string, number[]> {
    const vectorMap: Record<string, number[]> = {
        full: embedding
    }
    for (const dim of truncatedDimensions) {
        if (dim >= fullDimension) continue
        vectorMap[`${dim}`] = truncateAndNormalize(embedding, dim)
    }
    return vectorMap
}

/**
 * Ensure the MRL collection exists with named vectors.
 * If the collection already exists, this is a no-op.
 */
async function ensureMrlCollection(client: QdrantClient, collectionName: string, collectionConfig: Record<string, any>): Promise<void> {
    const response = await client.getCollections()
    const collectionNames = response.collections.map((c: any) => c.name)
    if (!collectionNames.includes(collectionName)) {
        await client.createCollection(collectionName, collectionConfig)
    }
}

class Qdrant_VectorStores implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    badge: string
    baseClasses: string[]
    inputs: INodeParams[]
    credential: INodeParams
    outputs: INodeOutputsValue[]

    constructor() {
        this.label = 'Qdrant'
        this.name = 'qdrant'
        this.version = 6.0
        this.type = 'Qdrant'
        this.icon = 'qdrant.png'
        this.category = 'Vector Stores'
        this.description =
            'Upsert embedded data and perform similarity search upon query using Qdrant, a scalable open source vector database written in Rust'
        this.baseClasses = [this.type, 'VectorStoreRetriever', 'BaseRetriever']
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            description: 'Only needed when using Qdrant cloud hosted',
            optional: true,
            credentialNames: ['qdrantApi']
        }
        this.inputs = [
            {
                label: 'Document',
                name: 'document',
                type: 'Document',
                list: true,
                optional: true
            },
            {
                label: 'Embeddings',
                name: 'embeddings',
                type: 'Embeddings'
            },
            {
                label: 'Record Manager',
                name: 'recordManager',
                type: 'RecordManager',
                description: 'Keep track of the record to prevent duplication',
                optional: true
            },
            {
                label: 'Qdrant Server URL',
                name: 'qdrantServerUrl',
                type: 'string',
                placeholder: 'http://localhost:6333'
            },
            {
                label: 'Qdrant Collection Name',
                name: 'qdrantCollection',
                type: 'string',
                acceptVariable: true
            },
            {
                label: 'File Upload',
                name: 'fileUpload',
                description: 'Allow file upload on the chat',
                hint: {
                    label: 'How to use',
                    value: howToUseFileUpload
                },
                type: 'boolean',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Vector Dimension',
                name: 'qdrantVectorDimension',
                type: 'number',
                default: 1536,
                additionalParams: true
            },
            {
                label: 'Content Key',
                name: 'contentPayloadKey',
                description: 'The key for storing text. Default to `content`',
                type: 'string',
                default: 'content',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Metadata Key',
                name: 'metadataPayloadKey',
                description: 'The key for storing metadata. Default to `metadata`',
                type: 'string',
                default: 'metadata',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Upsert Batch Size',
                name: 'batchSize',
                type: 'number',
                step: 1,
                description: 'Upsert in batches of size N',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Similarity',
                name: 'qdrantSimilarity',
                description: 'Similarity measure used in Qdrant.',
                type: 'options',
                default: 'Cosine',
                options: [
                    {
                        label: 'Cosine',
                        name: 'Cosine'
                    },
                    {
                        label: 'Euclid',
                        name: 'Euclid'
                    },
                    {
                        label: 'Dot',
                        name: 'Dot'
                    }
                ],
                additionalParams: true
            },
            {
                label: 'Enable MRL (Matryoshka Representation Learning)',
                name: 'mrlEnabled',
                description:
                    'Store embeddings at multiple truncated dimensions using Qdrant named vectors. Requires an MRL-trained embedding model (e.g. OpenAI text-embedding-3-*, Nomic Embed, jina-embeddings-v2).',
                type: 'boolean',
                default: false,
                additionalParams: true,
                optional: true
            },
            {
                label: 'MRL Truncated Dimensions',
                name: 'mrlDimensions',
                description:
                    'Comma-separated list of truncated dimensions to store alongside the full embedding. E.g. "512, 256, 128". Each must be smaller than the full Vector Dimension.',
                type: 'string',
                placeholder: '512, 256',
                additionalParams: true,
                optional: true
            },
            {
                label: 'MRL Search Vector',
                name: 'mrlSearchVector',
                description:
                    'Which named vector to use for similarity search. "full" uses the original embedding, or specify a dimension number (e.g. "256") to search the truncated vector. Smaller dimensions are faster but less precise.',
                type: 'string',
                default: 'full',
                placeholder: 'full',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Additional Collection Cofiguration',
                name: 'qdrantCollectionConfiguration',
                description:
                    'Refer to <a target="_blank" href="https://qdrant.tech/documentation/concepts/collections">collection docs</a> for more reference',
                type: 'json',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top K',
                name: 'topK',
                description: 'Number of top results to fetch. Default to 4',
                placeholder: '4',
                type: 'number',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Qdrant Search Filter',
                name: 'qdrantFilter',
                description: 'Only return points which satisfy the conditions',
                type: 'json',
                additionalParams: true,
                optional: true,
                acceptVariable: true
            }
        ]
        this.outputs = [
            {
                label: 'Qdrant Retriever',
                name: 'retriever',
                baseClasses: this.baseClasses
            },
            {
                label: 'Qdrant Vector Store',
                name: 'vectorStore',
                baseClasses: [this.type, ...getBaseClasses(QdrantVectorStore)]
            }
        ]
    }

    //@ts-ignore
    vectorStoreMethods = {
        async upsert(nodeData: INodeData, options: ICommonObject): Promise<Partial<IndexingResult>> {
            const qdrantServerUrl = nodeData.inputs?.qdrantServerUrl as string
            const collectionName = nodeData.inputs?.qdrantCollection as string
            const docs = nodeData.inputs?.document as Document[]
            const embeddings = nodeData.inputs?.embeddings as Embeddings
            const qdrantSimilarity = nodeData.inputs?.qdrantSimilarity
            const qdrantVectorDimension = nodeData.inputs?.qdrantVectorDimension
            const recordManager = nodeData.inputs?.recordManager
            const _batchSize = nodeData.inputs?.batchSize
            const contentPayloadKey = nodeData.inputs?.contentPayloadKey || 'content'
            const metadataPayloadKey = nodeData.inputs?.metadataPayloadKey || 'metadata'
            const isFileUploadEnabled = nodeData.inputs?.fileUpload as boolean

            const mrlEnabled = nodeData.inputs?.mrlEnabled as boolean
            const mrlDimensionsRaw = (nodeData.inputs?.mrlDimensions as string) || ''
            const mrlDimensions = mrlEnabled ? parseMrlDimensions(mrlDimensionsRaw) : []

            const credentialData = await getCredentialData(nodeData.credential ?? '', options)
            const qdrantApiKey = getCredentialParam('qdrantApiKey', credentialData, nodeData)

            const port = Qdrant_VectorStores.determinePortByUrl(qdrantServerUrl)

            const client = new QdrantClient({
                url: qdrantServerUrl,
                apiKey: qdrantApiKey,
                port: port
            })

            const flattenDocs = docs && docs.length ? flatten(docs) : []
            const finalDocs: Document[] = []
            for (let i = 0; i < flattenDocs.length; i += 1) {
                if (flattenDocs[i] && flattenDocs[i].pageContent) {
                    if (isFileUploadEnabled && options.chatId) {
                        flattenDocs[i].metadata = { ...flattenDocs[i].metadata, [FLOWISE_CHATID]: options.chatId }
                    }
                    finalDocs.push(new Document(flattenDocs[i]))
                }
            }

            const fullDimension = qdrantVectorDimension ? parseInt(qdrantVectorDimension, 10) : 1536
            const distance = qdrantSimilarity ?? 'Cosine'

            const collectionConfig =
                mrlEnabled && mrlDimensions.length > 0
                    ? { vectors: buildMrlVectorsConfig(fullDimension, mrlDimensions, distance) }
                    : { vectors: { size: fullDimension, distance } }

            const dbConfig: QdrantLibArgs = {
                client: client as any,
                url: qdrantServerUrl,
                collectionName,
                collectionConfig,
                contentPayloadKey,
                metadataPayloadKey
            }

            try {
                if (recordManager) {
                    const vectorStore = new QdrantVectorStore(embeddings, dbConfig)

                    if (mrlEnabled && mrlDimensions.length > 0) {
                        await ensureMrlCollection(client, collectionName, collectionConfig)
                    } else {
                        await vectorStore.ensureCollection()
                    }

                    vectorStore.addVectors = async (
                        vectors: number[][],
                        documents: Document[],
                        documentOptions?: QdrantAddDocumentOptions
                    ): Promise<void> => {
                        if (vectors.length === 0) {
                            return
                        }

                        if (mrlEnabled && mrlDimensions.length > 0) {
                            await ensureMrlCollection(client, collectionName, collectionConfig)
                        } else {
                            await vectorStore.ensureCollection()
                        }

                        const points = vectors.map((embedding, idx) => ({
                            id: documentOptions?.ids?.length ? documentOptions?.ids[idx] : uuid(),
                            vector:
                                mrlEnabled && mrlDimensions.length > 0
                                    ? buildMrlVectorMap(embedding, mrlDimensions, fullDimension)
                                    : embedding,
                            payload: {
                                [contentPayloadKey]: documents[idx].pageContent,
                                [metadataPayloadKey]: documents[idx].metadata,
                                customPayload: documentOptions?.customPayload?.length ? documentOptions?.customPayload[idx] : undefined
                            }
                        }))

                        try {
                            if (_batchSize) {
                                const batchSize = parseInt(_batchSize, 10)
                                for (let i = 0; i < points.length; i += batchSize) {
                                    const batchPoints = points.slice(i, i + batchSize)
                                    await client.upsert(collectionName, {
                                        wait: true,
                                        points: batchPoints as any
                                    })
                                }
                            } else {
                                await client.upsert(collectionName, {
                                    wait: true,
                                    points: points as any
                                })
                            }
                        } catch (e: any) {
                            const error = new Error(`${e?.status ?? 'Undefined error code'} ${e?.message}: ${e?.data?.status?.error}`)
                            throw error
                        }
                    }

                    vectorStore.delete = async (params: { ids: string[] }): Promise<void> => {
                        const { ids } = params

                        if (ids?.length) {
                            try {
                                client.delete(collectionName, {
                                    points: ids
                                })
                            } catch (e) {
                                console.error('Failed to delete')
                            }
                        }
                    }

                    await recordManager.createSchema()

                    const res = await index({
                        docsSource: finalDocs,
                        recordManager,
                        vectorStore,
                        options: {
                            cleanup: recordManager?.cleanup,
                            sourceIdKey: recordManager?.sourceIdKey ?? 'source',
                            vectorStoreName: collectionName
                        }
                    })

                    return res
                } else {
                    if (mrlEnabled && mrlDimensions.length > 0) {
                        await ensureMrlCollection(client, collectionName, collectionConfig)

                        const texts = finalDocs.map((doc) => doc.pageContent)
                        const allEmbeddings = await embeddings.embedDocuments(texts)

                        const points = allEmbeddings.map((embedding: number[], idx: number) => ({
                            id: uuid(),
                            vector: buildMrlVectorMap(embedding, mrlDimensions, fullDimension),
                            payload: {
                                [contentPayloadKey]: finalDocs[idx].pageContent,
                                [metadataPayloadKey]: finalDocs[idx].metadata
                            }
                        }))

                        if (_batchSize) {
                            const batchSize = parseInt(_batchSize, 10)
                            for (let i = 0; i < points.length; i += batchSize) {
                                const batchPoints = points.slice(i, i + batchSize)
                                await client.upsert(collectionName, {
                                    wait: true,
                                    points: batchPoints as any
                                })
                            }
                        } else {
                            await client.upsert(collectionName, {
                                wait: true,
                                points: points as any
                            })
                        }

                        return { numAdded: finalDocs.length, addedDocs: finalDocs }
                    } else {
                        if (_batchSize) {
                            const batchSize = parseInt(_batchSize, 10)
                            for (let i = 0; i < finalDocs.length; i += batchSize) {
                                const batch = finalDocs.slice(i, i + batchSize)
                                await QdrantVectorStore.fromDocuments(batch, embeddings, dbConfig)
                            }
                        } else {
                            await QdrantVectorStore.fromDocuments(finalDocs, embeddings, dbConfig)
                        }
                        return { numAdded: finalDocs.length, addedDocs: finalDocs }
                    }
                }
            } catch (e) {
                throw new Error(e)
            }
        },
        async delete(nodeData: INodeData, ids: string[], options: ICommonObject): Promise<void> {
            const qdrantServerUrl = nodeData.inputs?.qdrantServerUrl as string
            const collectionName = nodeData.inputs?.qdrantCollection as string
            const embeddings = nodeData.inputs?.embeddings as Embeddings
            const qdrantSimilarity = nodeData.inputs?.qdrantSimilarity
            const qdrantVectorDimension = nodeData.inputs?.qdrantVectorDimension
            const recordManager = nodeData.inputs?.recordManager

            const credentialData = await getCredentialData(nodeData.credential ?? '', options)
            const qdrantApiKey = getCredentialParam('qdrantApiKey', credentialData, nodeData)

            const port = Qdrant_VectorStores.determinePortByUrl(qdrantServerUrl)

            const client = new QdrantClient({
                url: qdrantServerUrl,
                apiKey: qdrantApiKey,
                port: port
            })

            const dbConfig: QdrantLibArgs = {
                client: client as any,
                url: qdrantServerUrl,
                collectionName,
                collectionConfig: {
                    vectors: {
                        size: qdrantVectorDimension ? parseInt(qdrantVectorDimension, 10) : 1536,
                        distance: qdrantSimilarity ?? 'Cosine'
                    }
                }
            }

            const vectorStore = new QdrantVectorStore(embeddings, dbConfig)

            vectorStore.delete = async (params: { ids: string[] }): Promise<void> => {
                const { ids } = params

                if (ids?.length) {
                    try {
                        client.delete(collectionName, {
                            points: ids
                        })
                    } catch (e) {
                        console.error('Failed to delete')
                    }
                }
            }

            try {
                if (recordManager) {
                    const vectorStoreName = collectionName
                    await recordManager.createSchema()
                    ;(recordManager as any).namespace = (recordManager as any).namespace + '_' + vectorStoreName
                    const filterKeys: ICommonObject = {}
                    if (options.docId) {
                        filterKeys.docId = options.docId
                    }
                    const keys: string[] = await recordManager.listKeys(filterKeys)

                    await vectorStore.delete({ ids: keys })
                    await recordManager.deleteKeys(keys)
                } else {
                    await vectorStore.delete({ ids })
                }
            } catch (e) {
                throw new Error(e)
            }
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const qdrantServerUrl = nodeData.inputs?.qdrantServerUrl as string
        const collectionName = nodeData.inputs?.qdrantCollection as string
        let qdrantCollectionConfiguration = nodeData.inputs?.qdrantCollectionConfiguration
        const embeddings = nodeData.inputs?.embeddings as Embeddings
        const qdrantSimilarity = nodeData.inputs?.qdrantSimilarity
        const qdrantVectorDimension = nodeData.inputs?.qdrantVectorDimension
        const output = nodeData.outputs?.output as string
        const topK = nodeData.inputs?.topK as string
        let queryFilter = nodeData.inputs?.qdrantFilter
        const contentPayloadKey = nodeData.inputs?.contentPayloadKey || 'content'
        const metadataPayloadKey = nodeData.inputs?.metadataPayloadKey || 'metadata'
        const isFileUploadEnabled = nodeData.inputs?.fileUpload as boolean

        const mrlEnabled = nodeData.inputs?.mrlEnabled as boolean
        const mrlSearchVector = (nodeData.inputs?.mrlSearchVector as string) || 'full'

        const k = topK ? parseFloat(topK) : 4

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const qdrantApiKey = getCredentialParam('qdrantApiKey', credentialData, nodeData)

        const port = Qdrant_VectorStores.determinePortByUrl(qdrantServerUrl)

        const client = new QdrantClient({
            url: qdrantServerUrl,
            apiKey: qdrantApiKey,
            port: port
        })

        const dbConfig: QdrantLibArgs = {
            client: client as any,
            collectionName,
            contentPayloadKey,
            metadataPayloadKey
        }

        const retrieverConfig: RetrieverConfig = {
            k
        }

        if (qdrantCollectionConfiguration) {
            qdrantCollectionConfiguration =
                typeof qdrantCollectionConfiguration === 'object'
                    ? qdrantCollectionConfiguration
                    : parseJsonBody(qdrantCollectionConfiguration)
            dbConfig.collectionConfig = {
                ...qdrantCollectionConfiguration,
                vectors: {
                    ...qdrantCollectionConfiguration.vectors,
                    size: qdrantVectorDimension ? parseInt(qdrantVectorDimension, 10) : 1536,
                    distance: qdrantSimilarity ?? 'Cosine'
                }
            }
        }

        if (queryFilter) {
            retrieverConfig.filter = typeof queryFilter === 'object' ? queryFilter : parseJsonBody(queryFilter)
        }
        if (isFileUploadEnabled && options.chatId) {
            retrieverConfig.filter = retrieverConfig.filter || {}

            retrieverConfig.filter.should = Array.isArray(retrieverConfig.filter.should) ? retrieverConfig.filter.should : []

            retrieverConfig.filter.should.push(
                {
                    key: `metadata.${FLOWISE_CHATID}`,
                    match: {
                        value: options.chatId
                    }
                },
                {
                    is_empty: {
                        key: `metadata.${FLOWISE_CHATID}`
                    }
                }
            )
        }

        const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, dbConfig)

        // For MRL: override similaritySearchVectorWithScore to use the `using` parameter
        if (mrlEnabled && mrlSearchVector) {
            const searchVectorName = mrlSearchVector
            const origFilter = retrieverConfig.filter

            vectorStore.similaritySearchVectorWithScore = async (
                query: number[],
                searchK: number,
                filter?: Record<string, any>
            ): Promise<[Document, number][]> => {
                const searchFilter = filter ?? origFilter

                // Truncate query vector if searching a smaller named vector
                let searchQuery: number[] = query
                if (searchVectorName !== 'full') {
                    const targetDim = parseInt(searchVectorName, 10)
                    if (!isNaN(targetDim) && query.length > targetDim) {
                        searchQuery = truncateAndNormalize(query, targetDim)
                    }
                }

                const results = (
                    await client.query(collectionName, {
                        query: searchQuery,
                        using: searchVectorName,
                        limit: searchK,
                        filter: searchFilter,
                        with_payload: [metadataPayloadKey, contentPayloadKey],
                        with_vector: false
                    })
                ).points

                return results.map((res: any) => [
                    new Document({
                        id: res.id,
                        metadata: res.payload?.[metadataPayloadKey],
                        pageContent: res.payload?.[contentPayloadKey] ?? ''
                    }),
                    res.score
                ])
            }
        }

        if (output === 'retriever') {
            const retriever = vectorStore.asRetriever(retrieverConfig)
            return retriever
        } else if (output === 'vectorStore') {
            ;(vectorStore as any).k = k
            if (queryFilter) {
                ;(vectorStore as any).filter = retrieverConfig.filter
            }
            return vectorStore
        }
        return vectorStore
    }

    /**
     * Determine the port number from the given URL.
     *
     * The problem is when not doing this the qdrant-client.js will fall back on 6663 when you enter a port 443 and 80.
     * See: https://stackoverflow.com/questions/59104197/nodejs-new-url-urlhttps-myurl-com80-lists-the-port-as-empty
     * @param qdrantServerUrl the url to get the port from
     */
    static determinePortByUrl(qdrantServerUrl: string): number {
        const parsedUrl = new URL(qdrantServerUrl)

        let port = parsedUrl.port ? parseInt(parsedUrl.port) : 6663

        if (parsedUrl.protocol === 'https:' && parsedUrl.port === '') {
            port = 443
        }
        if (parsedUrl.protocol === 'http:' && parsedUrl.port === '') {
            port = 80
        }

        return port
    }
}

module.exports = { nodeClass: Qdrant_VectorStores }
