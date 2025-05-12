package org.jetbrains.plugins

import dev.langchain4j.data.document.DocumentSplitter
import dev.langchain4j.data.document.splitter.DocumentSplitters
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.memory.chat.ChatMemoryProvider
import dev.langchain4j.memory.chat.MessageWindowChatMemory
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel
import dev.langchain4j.model.openai.OpenAiStreamingChatModel
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever
import dev.langchain4j.store.embedding.EmbeddingStore
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore
import io.ktor.server.application.Application
import kotlinx.serialization.Serializable
import org.jetbrains.ai.ExposedChatMemoryStore
import org.jetbrains.ai.LangChainFactory
import org.jetbrains.ai.TravelService
import org.jetbrains.exposed.sql.Database
import org.koin.core.module.dsl.singleOf
import org.koin.ktor.plugin.koinModule

@Serializable
data class AIConfig(
    val baseUrl: String,
    val apiKey: String,
    val model: String,
    val maxSegmentSizeInTokens: Int,
    val maxOverlapSizeInTokens: Int,
)

fun Application.aiModule(config: AIConfig) {
    koinModule {
        single { model() }
        single { embeddingStore() }
        single { chat(config) }
        singleOf(::retriever)
        single { splitter(config) }
        singleOf(::memory)
        singleOf(::ingestor)
        singleOf(::LangChainFactory)
        single { get<LangChainFactory>().service<TravelService>() }
    }
}

// Temp work around https://youtrack.jetbrains.com/issue/KTOR-8477/DI-provide-lambda-type-fails-for-java-function-returns
private fun embeddingStore(): EmbeddingStore<TextSegment> = InMemoryEmbeddingStore()

private fun chat(config: AIConfig): StreamingChatModel =
    OpenAiStreamingChatModel.builder()
        .apiKey(config.apiKey)
        .modelName(config.model)
        .build()

private fun model(): EmbeddingModel = AllMiniLmL6V2QuantizedEmbeddingModel()

private fun memory(db: Database): ChatMemoryProvider = ChatMemoryProvider { memoryId: Any? ->
    MessageWindowChatMemory.builder()
        .id(memoryId)
        .maxMessages(10)
        .chatMemoryStore(ExposedChatMemoryStore(db))
        .build()
}

private fun retriever(model: EmbeddingModel): EmbeddingStoreContentRetriever =
    EmbeddingStoreContentRetriever.builder()
        .embeddingStore(InMemoryEmbeddingStore())
        .embeddingModel(model)
        .maxResults(5)
        .minScore(0.5)
        .build()

private fun ingestor(
    store: EmbeddingStore<TextSegment>,
    model: EmbeddingModel,
    splitter: DocumentSplitter
): EmbeddingStoreIngestor = EmbeddingStoreIngestor.builder()
    .embeddingStore(store)
    .embeddingModel(model)
    .documentSplitter(splitter)
    .build()

private fun splitter(config: AIConfig): DocumentSplitter =
    DocumentSplitters.recursive(config.maxSegmentSizeInTokens, config.maxOverlapSizeInTokens)
