# Folder overview layout

Folder overviews start below the navigation bar and use a wider content column (up to 72rem).
The composer, title and chat list share the available width. The page scrolls naturally,
so tall screens display more rows without increasing text density. New-chat screens retain
their existing centered layout.

The overview requests 50 chats per page. Changing pages returns to the top of the list.
The shared-folder endpoint accepts `page_size` from 1 to 100 when `page` is supplied;
its default remains 10, and requests without `page` retain the existing 60-chat limit.
Sorting, ownership and shared-folder access checks are unchanged.
