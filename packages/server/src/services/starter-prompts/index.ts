import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'
import { getErrorMessage } from '../../errors/utils'
import { getRunningExpressApp } from '../../utils/getRunningExpressApp'
import { databaseEntities, CURRENT_DATE_TIME_VAR_PREFIX } from '../../utils'
import { ChatFlow } from '../../database/entities/ChatFlow'
import { generateFollowUpPrompts, FollowUpPromptConfig, ICommonObject } from 'flowise-components'
import { QdrantClient } from '@qdrant/js-client-rest'
import { v4 as uuidv4 } from 'uuid'
import logger from '../../utils/logger'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface MetadataField {
    key: string
    sourceVar: string
    type: 'string' | 'number' | 'array'
    useAsFilter: boolean
}

interface QdrantConfig {
    enabled: boolean
    metadataOnly?: boolean
    maxArticles?: number
    qdrantServerUrl: string
    qdrantCredentialId: string
    collectionName: string
    vectorDimension: number
    embeddingCredentialId: string
    embeddingModelName: string
    embeddingBasePath?: string
    metadataFields: MetadataField[]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Determine the port number from the given URL.
 * Copied from Qdrant node (Qdrant.ts) — the qdrant-client defaults to 6663
 * when the URL omits the port for standard HTTP/HTTPS schemes.
 */
function determinePort(url: string): number {
    const parsedUrl = new URL(url)
    let port = parsedUrl.port ? parseInt(parsedUrl.port) : 6663
    if (parsedUrl.protocol === 'https:' && parsedUrl.port === '') port = 443
    if (parsedUrl.protocol === 'http:' && parsedUrl.port === '') port = 80
    return port
}

/**
 * Create an OpenAI Custom Embeddings instance using the same pattern as
 * documentstore/index.ts _createEmbeddingsObject.
 */
async function createEmbeddingInstance(qdrantConfig: QdrantConfig, appServer: any): Promise<any> {
    const componentNodes = appServer.nodesPool.componentNodes
    const embeddingComponent = componentNodes['openAIEmbeddingsCustom']
    if (!embeddingComponent) {
        throw new InternalFlowiseError(
            StatusCodes.INTERNAL_SERVER_ERROR,
            'OpenAI Custom Embeddings component not found in node pool'
        )
    }

    const embeddingNodeData: any = {
        inputs: {
            modelName: qdrantConfig.embeddingModelName || '',
            basepath: qdrantConfig.embeddingBasePath || ''
        },
        outputs: { output: 'document' },
        id: `${embeddingComponent.name}_0`,
        label: embeddingComponent.label,
        name: embeddingComponent.name,
        category: embeddingComponent.category,
        inputParams: embeddingComponent.inputs || []
    }
    if (qdrantConfig.embeddingCredentialId) {
        embeddingNodeData.credential = qdrantConfig.embeddingCredentialId
    }

    const options: ICommonObject = {
        appDataSource: appServer.AppDataSource,
        databaseEntities,
        logger
    }

    const embeddingNodeModule = await import(embeddingComponent.filePath as string)
    const embeddingNodeInstance = new embeddingNodeModule.nodeClass()
    const embeddingObj = await embeddingNodeInstance.init(embeddingNodeData, '', options)
    if (!embeddingObj) {
        throw new InternalFlowiseError(StatusCodes.INTERNAL_SERVER_ERROR, 'Failed to create embedding instance')
    }
    return embeddingObj
}

/**
 * Create a QdrantClient with the configured URL and optional API key.
 */
async function createQdrantClient(qdrantConfig: QdrantConfig, appServer: any): Promise<QdrantClient> {
    let apiKey: string | undefined
    if (qdrantConfig.qdrantCredentialId) {
        const { getCredentialData } = await import('flowise-components')
        const credentialData = await getCredentialData(qdrantConfig.qdrantCredentialId, {
            appDataSource: appServer.AppDataSource,
            databaseEntities
        })
        apiKey = credentialData?.qdrantApiKey
    }

    const port = determinePort(qdrantConfig.qdrantServerUrl)
    return new QdrantClient({
        url: qdrantConfig.qdrantServerUrl,
        apiKey,
        port
    })
}

/**
 * Ensure the collection exists. Create it with the configured vector dimension if missing.
 */
async function ensureCollection(client: QdrantClient, collectionName: string, vectorDimension: number): Promise<void> {
    const response = await client.getCollections()
    const names = response.collections.map((c: any) => c.name)
    if (!names.includes(collectionName)) {
        await client.createCollection(collectionName, {
            vectors: {
                size: vectorDimension,
                distance: 'Cosine'
            }
        })
    }
}

/**
 * Build Qdrant filter `must` clauses from metadata fields + runtime vars.
 */
function buildQdrantFilter(metadataFields: MetadataField[], vars: Record<string, any>): Record<string, any> | null {
    const must: any[] = []
    for (const field of metadataFields) {
        if (!field.useAsFilter) continue
        const value = vars[field.sourceVar]
        if (value === undefined || value === null || value === '') continue

        if (field.type === 'array' && Array.isArray(value)) {
            // Match any of the values in the array
            must.push({ key: `metadata.${field.key}`, match: { any: value } })
        } else {
            must.push({ key: `metadata.${field.key}`, match: { value } })
        }
    }
    return must.length > 0 ? { must } : null
}

/**
 * Build the metadata payload object from metadata fields + runtime vars.
 */
function buildMetadataPayload(metadataFields: MetadataField[], vars: Record<string, any>): Record<string, any> {
    const metadata: Record<string, any> = {}
    for (const field of metadataFields) {
        const value = vars[field.sourceVar]
        if (value === undefined || value === null) continue
        if (field.type === 'number') {
            metadata[field.key] = typeof value === 'number' ? value : parseFloat(value)
        } else {
            metadata[field.key] = value
        }
    }
    return metadata
}

/**
 * Extract article IDs from the mobile app's qdrantFilter.
 * The mobile app sends: { must: [{ has_id: [479264, 479246, ...] }] }
 */
function extractArticleIds(overrideConfig: Record<string, any>): (string | number)[] {
    const qdrantFilter = overrideConfig.qdrantFilter
    if (!qdrantFilter?.must || !Array.isArray(qdrantFilter.must)) return []
    for (const clause of qdrantFilter.must) {
        if (clause.has_id && Array.isArray(clause.has_id)) {
            return clause.has_id
        }
    }
    return []
}

/**
 * Direct ID lookup against the Retrieve collection: pull each article's stored text by its
 * Qdrant point ID. No similarity, no embedding, no metadata filter — just `client.retrieve`.
 * Used to feed article content into the LLM via the {retrieved_from_vector_db} prompt variable.
 *
 * Tries common payload field names so this works against collections built by Flowise document
 * loaders (`pageContent`), LangChain Python (`page_content`), or hand-rolled (`content`/`text`).
 */
async function retrieveArticleTextsByIds(qdrantConfig: QdrantConfig, articleIds: (string | number)[], appServer: any): Promise<string[]> {
    if (articleIds.length === 0) return []
    const client = await createQdrantClient(qdrantConfig, appServer)
    const points = (await client.retrieve(qdrantConfig.collectionName, {
        ids: articleIds as any,
        with_payload: true,
        with_vector: false
    })) as any[]
    const texts: string[] = []
    for (const point of points || []) {
        const payload = point?.payload || {}
        const text = payload.content ?? payload.pageContent ?? payload.page_content ?? payload.text
        if (typeof text === 'string' && text.trim()) texts.push(text)
    }
    return texts
}

// ---------------------------------------------------------------------------
// Qdrant Retrieve
// ---------------------------------------------------------------------------

async function retrieveFromQdrant(
    qdrantConfig: QdrantConfig,
    overrideConfig: Record<string, any>,
    appServer: any
): Promise<{ questions: string[]; qdrantIds: string[] } | null> {
    const vars = overrideConfig.vars || {}
    const metadataFilter = buildQdrantFilter(qdrantConfig.metadataFields || [], vars)

    // Metadata-only path: exact-match cache lookup, no embedding, no similarity ranking.
    // We deliberately skip the article_ids overlap clause here — articles in a cluster can
    // change, but the cache key (e.g. cluster_id) should be stable, so adding article_ids
    // to the filter would force regeneration on every article change.
    if (qdrantConfig.metadataOnly) {
        if (!metadataFilter) return null

        const client = await createQdrantClient(qdrantConfig, appServer)
        const results = await client.scroll(qdrantConfig.collectionName, {
            filter: metadataFilter,
            limit: 4,
            with_payload: true,
            with_vector: false
        })

        const points = (results as any).points || []
        if (!points || points.length === 0) return null

        const questions: string[] = []
        const qdrantIds: string[] = []
        for (const point of points) {
            const content = point.payload?.content
            if (content && typeof content === 'string') {
                questions.push(content)
                qdrantIds.push(String(point.id))
            }
        }

        return questions.length > 0 ? { questions, qdrantIds } : null
    }

    // Vector-similarity path (original behaviour): metadata filter + article_ids overlap +
    // similarity rank against the request's story_content.
    const articleIds = extractArticleIds(overrideConfig)
    const must: any[] = metadataFilter?.must || []
    if (articleIds.length > 0) {
        must.push({ key: 'metadata.article_ids', match: { any: articleIds.map(String) } })
    }

    if (must.length === 0) return null // no filter criteria → can't retrieve meaningfully

    const filter = { must }

    const embeddings = await createEmbeddingInstance(qdrantConfig, appServer)
    const client = await createQdrantClient(qdrantConfig, appServer)

    // Build a query text from available context
    const queryText = vars.story_content
        ? (vars.story_content as string).slice(0, 500)
        : Object.values(vars).filter((v) => typeof v === 'string').join(' ')
    if (!queryText.trim()) return null

    const queryVector = await embeddings.embedQuery(queryText)

    const results = await client.query(qdrantConfig.collectionName, {
        query: queryVector,
        limit: 4,
        filter,
        with_payload: true,
        with_vector: false
    })

    const points = (results as any).points || results
    if (!points || points.length === 0) return null

    const questions: string[] = []
    const qdrantIds: string[] = []
    for (const point of points) {
        const content = point.payload?.content
        if (content && typeof content === 'string') {
            questions.push(content)
            qdrantIds.push(String(point.id))
        }
    }

    return questions.length > 0 ? { questions, qdrantIds } : null
}

// ---------------------------------------------------------------------------
// Qdrant Store (non-blocking)
// ---------------------------------------------------------------------------

function storeToQdrant(
    qdrantConfig: QdrantConfig,
    questions: string[],
    overrideConfig: Record<string, any>,
    appServer: any
): string[] {
    // Pre-generate UUIDs so we can return them immediately
    const ids = questions.map(() => uuidv4())

    // Fire-and-forget — the caller returns prompts to the user right away
    ;(async () => {
        try {
            const embeddings = await createEmbeddingInstance(qdrantConfig, appServer)
            const client = await createQdrantClient(qdrantConfig, appServer)

            await ensureCollection(client, qdrantConfig.collectionName, qdrantConfig.vectorDimension || 1536)

            const vectors = await embeddings.embedDocuments(questions)
            const vars = { ...(overrideConfig.vars || {}), [CURRENT_DATE_TIME_VAR_PREFIX]: new Date().toISOString() }
            const metadata = buildMetadataPayload(qdrantConfig.metadataFields || [], vars)

            // Attach article IDs from qdrantFilter so they're searchable on retrieve
            const articleIds = extractArticleIds(overrideConfig)
            if (articleIds.length > 0) {
                metadata.article_ids = articleIds.map(String)
            }

            const points = questions.map((question: string, idx: number) => ({
                id: ids[idx],
                vector: vectors[idx],
                payload: {
                    content: question,
                    metadata
                }
            }))

            await client.upsert(qdrantConfig.collectionName, {
                wait: true,
                points: points as any
            })

            logger.info(`[StarterPrompts] Stored ${points.length} prompts to Qdrant collection "${qdrantConfig.collectionName}"`)
        } catch (err) {
            logger.error(`[StarterPrompts] Failed to store prompts to Qdrant: ${getErrorMessage(err)}`)
        }
    })()

    return ids
}

// ---------------------------------------------------------------------------
// Context builder (existing)
// ---------------------------------------------------------------------------

function buildContext(overrideConfig: Record<string, any>, flowData: string, chatflowName: string): string {
    const parts: string[] = []
    const vars = overrideConfig.vars || {}
    if (vars.story_content) {
        parts.push(`Topic content:\n${vars.story_content}`)
    }
    parts.push(`Chatflow: ${chatflowName}`)
    try {
        const flow = JSON.parse(flowData)
        if (flow.nodes && Array.isArray(flow.nodes)) {
            for (const node of flow.nodes) {
                const inputs = node.data?.inputs || {}
                for (const key of ['systemMessage', 'systemMessagePrompt', 'instructions']) {
                    if (inputs[key] && typeof inputs[key] === 'string' && inputs[key].trim()) {
                        parts.push(`System message: ${inputs[key].trim()}`)
                        break
                    }
                }
            }
        }
    } catch {
        // flowData parsing failed
    }
    const result = parts.join('\n\n')
    return result.length > 3000 ? result.slice(0, 3000) : result
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

const generateStarterPrompts = async (chatflowId: string, overrideConfig: Record<string, any>) => {
    try {
        const appServer = getRunningExpressApp()
        const chatflow = await appServer.AppDataSource.getRepository(ChatFlow).findOneBy({ id: chatflowId })
        if (!chatflow) {
            throw new InternalFlowiseError(StatusCodes.NOT_FOUND, `Chatflow ${chatflowId} not found`)
        }

        if (!chatflow.chatbotConfig) {
            throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'Chatbot config is not set')
        }

        let config: any
        try {
            config = JSON.parse(chatflow.chatbotConfig)
        } catch {
            throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'Failed to parse chatbot config')
        }

        const starterAiConfig = config.starterPrompts?.aiConfig
        const qdrantRaw = config.starterPrompts?.qdrantConfig || {}
        const retrieveConfig: QdrantConfig | undefined = qdrantRaw.retrieve
        const storeConfig: QdrantConfig | undefined = qdrantRaw.store

        // The prompt template is needed up front so we can decide whether to use Retrieve as
        // a cache shortcut or as a RAG inspiration source for the LLM.
        const provider = starterAiConfig?.selectedProvider
        const providerConfig = (provider && starterAiConfig?.[provider]) || {}
        const promptTemplate: string =
            providerConfig.prompt ||
            'Based on the following context, generate 4 short starter prompts a user might ask when first opening the chat. Each should be concise (under 100 characters), written from the user\'s perspective, and demonstrate different aspects of what this chatbot can help with.\n\nContext:\n{context}'
        const usesRetrievedVar = promptTemplate.includes('{retrieved_from_vector_db}')

        // ----- Step 1: Try retrieving from Qdrant -----
        // Two distinct modes, selected by whether the prompt template references the variable:
        //   • Variable in prompt → RAG mode: fetch article texts directly by ID from
        //     `overrideConfig.qdrantFilter.must[].has_id`, inject as {retrieved_from_vector_db},
        //     always proceed to the LLM. No cache shortcut.
        //   • Variable not in prompt → cache mode: existing similarity / metadata-only retrieve;
        //     on a hit, return the cached prompts and skip the LLM entirely.
        let retrievedContent = ''
        if (retrieveConfig?.enabled && retrieveConfig.qdrantServerUrl && retrieveConfig.collectionName) {
            if (usesRetrievedVar) {
                const allArticleIds = extractArticleIds(overrideConfig)
                // Cap to first k IDs (in client-supplied order) when maxArticles is a positive number.
                // Empty / 0 / undefined → no cap.
                const k = retrieveConfig.maxArticles
                const articleIds = typeof k === 'number' && k > 0 ? allArticleIds.slice(0, k) : allArticleIds
                if (articleIds.length > 0) {
                    try {
                        const texts = await retrieveArticleTextsByIds(retrieveConfig, articleIds, appServer)
                        retrievedContent = texts.join('\n\n')
                        const capNote = articleIds.length < allArticleIds.length ? ` (capped from ${allArticleIds.length} by k=${k})` : ''
                        logger.info(
                            `[StarterPrompts] RAG mode: fetched ${texts.length}/${articleIds.length} article texts by ID for {retrieved_from_vector_db}${capNote}`
                        )
                    } catch (err) {
                        logger.warn(`[StarterPrompts] Article-by-ID retrieve failed: ${getErrorMessage(err)}`)
                    }
                } else {
                    logger.info('[StarterPrompts] RAG mode: no article IDs in qdrantFilter.must[].has_id — variable will be empty')
                }
            } else {
                try {
                    const cached = await retrieveFromQdrant(retrieveConfig, overrideConfig, appServer)
                    if (cached) {
                        logger.info(`[StarterPrompts] Returned ${cached.questions.length} cached prompts from Qdrant`)
                        return cached
                    }
                } catch (err) {
                    logger.warn(`[StarterPrompts] Qdrant retrieve failed, falling back to LLM: ${getErrorMessage(err)}`)
                }
            }
        }

        // ----- Step 2: Generate via LLM -----
        if (!starterAiConfig || !provider) {
            throw new InternalFlowiseError(
                StatusCodes.BAD_REQUEST,
                'AI Starter Prompts must be configured with a provider before generating'
            )
        }

        const context = buildContext(overrideConfig, chatflow.flowData, chatflow.name)
        const finalPrompt = promptTemplate.replace('{context}', context).replace('{retrieved_from_vector_db}', retrievedContent)

        const starterConfig: FollowUpPromptConfig = {
            status: true,
            selectedProvider: provider,
            [provider]: {
                ...providerConfig,
                prompt: finalPrompt
            },
            skipWhenExhausted: false,
            deduplicationEnabled: false
        } as any

        const result = await generateFollowUpPrompts(starterConfig, '', {
            chatId: '',
            chatflowid: chatflowId,
            appDataSource: appServer.AppDataSource,
            databaseEntities: databaseEntities,
            question: '',
            sourceDocuments: '',
            chatHistory: '',
            analytic: chatflow.analytic || ''
        })

        const questions = result?.questions || []
        if (questions.length === 0) {
            return { questions: [] }
        }

        // ----- Step 3: Store to Qdrant (non-blocking) -----
        let qdrantIds: string[] | undefined
        if (storeConfig?.enabled && storeConfig.qdrantServerUrl && storeConfig.collectionName) {
            qdrantIds = storeToQdrant(storeConfig, questions, overrideConfig, appServer)
        }

        return qdrantIds ? { questions, qdrantIds } : { questions }
    } catch (error) {
        if (error instanceof InternalFlowiseError) throw error
        throw new InternalFlowiseError(StatusCodes.INTERNAL_SERVER_ERROR, `Error generating starter prompts: ${getErrorMessage(error)}`)
    }
}

export default {
    generateStarterPrompts
}
