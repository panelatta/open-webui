# Chat manager

Open **Manage Chats** below Search in the sidebar, or visit `/chats`.
The page manages only the signed-in user's conversations, including pinned and
archived chats. Internal agent chats are excluded.

- Search by title, filter by archive status or folder, and browse 50 rows per page.
- Click anywhere on a row (including its title) to select or deselect it, or use its
  checkbox. Enter/Space also toggles a focused row. The arrow at the end opens the chat
  without selecting it. Changing filters or pages clears selection.
- Move selected chats into a folder or out of folders, archive, unarchive, or delete.
- Each operation confirms the selected count. Delete is irreversible and stops active
  generation. Archiving also stops active generation. Both archive transitions
  follow the existing behavior of moving affected chats out of folders.
- Moving follows the existing behavior: it clears pinning, and moving into a folder
  unarchives the chat.
- Failed items are reported individually and remain selected when visible. Successful
  items are not retried. A network error prompts a refresh and review because the
  server may already have applied some changes.
- Desktop, mobile, light/dark mode, and English/Simplified Chinese are supported.

## API and authorization

`GET /api/v1/chats/manage` returns paginated metadata and a total. It supports
`query`, `page`, optional `archived`, and optional `folder_id` (empty means
no folder). It never returns message bodies.

`POST /api/v1/chats/bulk` accepts `ids` (1–50), `action`
(`move`, `archive`, `unarchive`, `delete`) and optional `folder_id`.
It deduplicates IDs, checks ownership even for administrators, checks delete and
folder permissions, and delegates to the existing single-chat operations for task
cancellation, tag cleanup, internal child cleanup and update events. Archive actions
specify a desired state so an already archived/unarchived chat is left as-is.
The response contains a success/error result for each unique ID.

No database migration, embedding change, provider setting, or deployment
configuration change is required.

## Validation

`backend/open_webui/test/utils/test_chat_manager.py` covers pagination, title
search escaping, private/internal chat exclusion, metadata-only results, permissions,
partial failures, desired archive states, batch limits and route validation.
Browser checks should use disposable chats in an isolated instance and cover batch
move, archive/unarchive, deletion cancellation/confirmation, filter and page selection
reset, partial failures, sidebar refresh, and mobile layout.
