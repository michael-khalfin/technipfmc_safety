import asyncio
import logging

# 导入我们找到的那个高级别的、封装好的 run 函数
from graphrag.index.run import run

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# 1. 将我们的主要逻辑封装在一个异步函数中
async def main():
    log.info("--- 开始执行GraphRAG索引流程 ---")
    
    # 2. 调用 run 函数。它会返回一个异步的可迭代对象，所以我们用 async for 来遍历
    # 这个函数的参数就是CLI的参数，非常简单直观
    async for result in run(
        root=".", # 对应CLI的 --root .
        config="config.yml", # 对应CLI的 --config config.yml
        # verbose=True, # 如果需要更详细的日志，可以加上这个
        # resume=None, # 如果需要从某个步骤恢复，可以设置
    ):
        log.info(f"工作流 '{result.workflow}' 已完成。")
        if result.errors:
            log.error(f"工作流 '{result.workflow}' 出现错误: {result.errors}")
        
    log.info("--- 所有工作流执行完毕 ---")


# 3. 在脚本的最后，使用 asyncio.run() 来启动并运行我们的 main 异步函数
if __name__ == "__main__":
    asyncio.run(main())
    