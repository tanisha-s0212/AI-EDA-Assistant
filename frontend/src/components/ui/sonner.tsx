"use client"

import { useTheme } from "next-themes"
import { Toaster as Sonner, ToasterProps } from "sonner"

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme()

  return (
    <Sonner
      {...props}
      theme={theme as ToasterProps["theme"]}
      position="bottom-left"
      className="toaster group"
      toastOptions={{
        ...props.toastOptions,
        style: {
          ...props.toastOptions?.style,
          maxWidth: "320px",
        },
      }}
      style={
        {
          ...props.style,
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
        } as React.CSSProperties
      }
    />
  )
}

export { Toaster }
