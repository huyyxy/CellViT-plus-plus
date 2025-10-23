# OpenLayers + Vite

此示例演示如何将 `ol` 包与 [Vite](https://vitejs.dev/) 一起使用。

要开始使用，请运行以下命令（需要Node 14+）:

    npx create-ol-app my-app --template vite

然后切换到你的新 `my-app` 目录并启动开发服务器（可在 http://localhost:5173 访问）:

    cd my-app
    npm start

要生成准备用于生产的构建:

    npm run build

然后将 `dist` 目录的内容部署到你的服务器。你也可以运行 `npm run serve` 来提供 `dist` 目录的结果以进行预览。
