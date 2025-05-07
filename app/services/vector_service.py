import logging
import os
from app import db
from app.models import Application, File
from app.adapters.qdrant_adapter import QdrantAdapter
from flask import current_app

logger = logging.getLogger(__name__)


def get_qdrant_adapter(use_reranker=False, use_semantic_chunking=None):
    """
    Создает и возвращает адаптер для Qdrant

    Args:
        use_reranker: Использовать ли ререйтинг
        use_semantic_chunking: Использовать ли семантическое разделение (None - брать из конфига)

    Returns:
        QdrantAdapter: Адаптер для Qdrant
    """
    # Если semantic_chunking не указано, берем из конфигурации
    if use_semantic_chunking is None:
        use_semantic_chunking = current_app.config.get('USE_SEMANTIC_CHUNKING', True)

    # Импортируем класс для получения настроек по умолчанию
    from ppee_analyzer.vector_store.ollama_embeddings import OllamaEmbeddings

    # Получаем настройки по умолчанию
    ollama_options = OllamaEmbeddings.get_default_options()

    # Можно переопределить некоторые настройки при необходимости
    # Например, для оптимизации производительности:
    if current_app.config.get('OPTIMIZE_EMBEDDINGS', False):
        ollama_options["num_thread"] = 12  # Увеличиваем число потоков

    return QdrantAdapter(
        host=current_app.config['QDRANT_HOST'],
        port=current_app.config['QDRANT_PORT'],
        collection_name=current_app.config['QDRANT_COLLECTION'],
        embeddings_type='ollama',
        model_name='bge-m3',
        ollama_url=current_app.config['OLLAMA_URL'],
        use_reranker=use_reranker,
        reranker_model='BAAI/bge-reranker-v2-m3',
        use_semantic_chunking=use_semantic_chunking,
        ollama_options=ollama_options
    )


def index_document(application_id, file_id, progress_callback=None):
    """
    Индексирует документ в векторной базе данных.

    Args:
        application_id: ID заявки
        file_id: ID файла
        progress_callback: Функция обратного вызова для обновления прогресса

    Returns:
        dict: Результат индексации
    """
    logger.info(f"Индексация документа для заявки {application_id}, файл {file_id}")

    # Получаем данные из БД
    file = File.query.get(file_id)
    if not file:
        error_msg = f"Файл с ID {file_id} не найден"
        logger.error(error_msg)
        raise ValueError(error_msg)

    application = Application.query.get(application_id)
    if not application:
        error_msg = f"Заявка с ID {application_id} не найдена"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Проверяем существование файла
    if not os.path.exists(file.file_path):
        error_msg = f"Файл не найден по пути: {file.file_path}"
        logger.error(error_msg)
        # Обновляем статус заявки
        application.status = "error"
        application.status_message = f"Файл не найден: {file.original_filename}"
        db.session.commit()
        raise FileNotFoundError(error_msg)

    # Обновляем статус
    if progress_callback:
        progress_callback(10, 'prepare', 'Проверка файла...')

    logger.info(f"Файл найден: {file.file_path} (размер: {os.path.getsize(file.file_path)} байт)")

    try:
        # Определяем, использовать ли семантическое разделение
        use_semantic_chunking = True

        # Проверяем расширение файла
        _, ext = os.path.splitext(file.file_path)
        ext = ext.lower()

        # Семантическое разделение имеет смысл только для PDF файлов
        if ext != '.pdf':
            use_semantic_chunking = False

        logger.info(f"Тип файла: {ext.lower()}, использование семантического разделения: {use_semantic_chunking}")

        # Получаем адаптер Qdrant с настройкой семантического разделения
        qdrant_adapter = get_qdrant_adapter(use_semantic_chunking=use_semantic_chunking)

        # Обновляем статус
        if progress_callback:
            progress_callback(15, 'convert', 'Конвертация документа...')

        # Индексируем документ с отслеживанием прогресса
        result = qdrant_adapter.index_document_with_progress(
            application_id=str(application_id),
            document_path=file.file_path,
            delete_existing=True,
            progress_callback=progress_callback
        )

        # Обновляем статус заявки на основе результата
        if result["status"] == "success":
            application.status = "indexed"
            logger.info(f"Индексация для заявки {application_id} успешно завершена")
        else:
            application.status = "error"
            application.status_message = result.get("error", "Неизвестная ошибка")
            logger.error(f"Ошибка при индексации заявки {application_id}: {result.get('error', 'Неизвестная ошибка')}")

        db.session.commit()
        return result

    except Exception as e:
        logger.exception(f"Ошибка при индексации документа: {str(e)}")

        # Обновляем статус заявки
        application.status = "error"
        application.status_message = str(e)
        db.session.commit()

        raise


def search(application_id, query, limit=5, use_reranker=False, rerank_limit=None,
           use_smart_search=False, vector_weight=0.5, text_weight=0.5, hybrid_threshold=10):
    """
    Выполняет семантический поиск.

    Args:
        application_id: ID заявки
        query: Поисковый запрос
        limit: Максимальное количество результатов
        use_reranker: Использовать ли ререйтинг
        rerank_limit: Количество документов для ререйтинга
        use_smart_search: Использовать ли умный выбор метода поиска
        vector_weight: Вес векторного поиска для гибридного
        text_weight: Вес текстового поиска для гибридного
        hybrid_threshold: Порог длины запроса для гибридного поиска

    Returns:
        list: Результаты поиска
    """
    if use_smart_search:
        logger.info(f"Выполнение умного поиска '{query}' для заявки {application_id} "
                   f"(ререйтинг: {use_reranker}, порог: {hybrid_threshold})")
    else:
        logger.info(f"Выполнение поиска '{query}' для заявки {application_id} (ререйтинг: {use_reranker})")

    try:
        # Получаем адаптер Qdrant
        qdrant_adapter = get_qdrant_adapter(use_reranker=use_reranker)

        # Выполняем поиск
        if use_smart_search:
            results = qdrant_adapter.smart_search(
                application_id=str(application_id),
                query=query,
                limit=limit,
                use_reranker=use_reranker,  # Явно передаем параметр use_reranker
                rerank_limit=rerank_limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                hybrid_threshold=hybrid_threshold
            )
        else:
            results = qdrant_adapter.search(
                application_id=str(application_id),
                query=query,
                limit=limit,
                rerank_limit=rerank_limit,
                use_reranker=use_reranker  # Явно передаем параметр use_reranker
            )

        # Освобождаем ресурсы, если использовался ререйтинг
        if use_reranker:
            try:
                qdrant_adapter.cleanup()
            except Exception as e:
                logger.warning(f"Ошибка при освобождении ресурсов ререйтинга: {str(e)}")

        return results

    except Exception as e:
        logger.exception(f"Ошибка при выполнении поиска: {str(e)}")

        # Освобождаем ресурсы в случае ошибки
        if 'qdrant_adapter' in locals() and qdrant_adapter.use_reranker:
            try:
                qdrant_adapter.cleanup()
            except:
                pass
        raise


def delete_application_data(application_id):
    """
    Удаляет данные заявки из векторной базы данных.

    Args:
        application_id: ID заявки

    Returns:
        bool: Успешность операции
    """
    logger.info(f"Удаление данных заявки {application_id} из векторной базы данных")

    try:
        # Получаем адаптер Qdrant
        qdrant_adapter = get_qdrant_adapter()

        # Удаляем данные
        success = qdrant_adapter.delete_application_data(str(application_id))

        return success

    except Exception as e:
        logger.exception(f"Ошибка при удалении данных заявки: {str(e)}")
        raise