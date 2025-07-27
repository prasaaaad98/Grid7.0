"use client"

import { useState, useEffect } from "react"
import { Star, Heart } from "lucide-react"

interface Product {
  id: number
  title: string
  brand: string
  category: string
  price: number
  retail_price: number
  images: string[]
  rating: number
  description: string
}

interface ResultsGridProps {
  query: string
  filters: any
  sort: string
  userLat: number | null
  userLon: number | null
}

export default function ResultsGrid({ query, filters, sort, userLat, userLon }: ResultsGridProps) {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!query.trim()) {
      setProducts([])
      return
    }

    setLoading(true)

    // Real backend call
    const searchParams = new URLSearchParams({
      q: query,
      min_price: filters.priceMin.toString(),
      max_price: filters.priceMax.toString(),
      min_rating: filters.rating.toString(),
      brand: filters.brands.join(','),
      category: filters.categories.join(',')
    })

    fetch(`http://localhost:8000/search?${searchParams}`)
      .then(res => res.json())
      .then(data => {
        // Transform backend data to match frontend Product interface
        const transformedProducts: Product[] = data.map((item: any, index: number) => ({
          id: item.id || index + 1,
          title: item.title,
          brand: item.brand,
          category: item.category,
          price: item.price,
          retail_price: item.retail_price,
          images: item.images,
          rating: item.rating,
          description: item.description,
        }))
        setProducts(transformedProducts)
      })
      .catch(error => {
        console.error('Error fetching search results:', error)
        // Fallback to dummy data if backend is not available
    setTimeout(() => {
      const dummyProducts: Product[] = [
        {
          id: 1,
          title: `${query} Premium Edition`,
          brand: "Apple",
          category: "Electronics",
          price: 25999,
          retail_price: 30000,
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query + " premium")],
          rating: 4.5,
          description: "High quality Apple product with premium features and excellent build quality",
        },
        {
          id: 2,
          title: `${query} Standard`,
          brand: "Samsung",
          category: "Electronics",
          price: 15999,
          retail_price: 18000,
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query + " standard")],
          rating: 4.2,
          description: "Reliable Samsung product with good features at affordable price",
        },
        {
          id: 3,
          title: `${query} Pro Max`,
          brand: "Sony",
          category: "Electronics",
          price: 35999,
          retail_price: 40000,
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query + " pro")],
          rating: 4.7,
          description: "Professional grade Sony product with advanced features",
        },
        {
          id: 4,
          title: `${query} Lite`,
          brand: "Realme",
          category: "Electronics",
          price: 8999,
          retail_price: 10000,
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query + " lite")],
          rating: 3.9,
          description: "Budget-friendly Realme product with essential features",
        },
      ]
      setProducts(dummyProducts)
    }, 500)
      })
      .finally(() => setLoading(false))
  }, [query, filters, sort, userLat, userLon])

  const addToCart = (product: Product) => {
    const cart = JSON.parse(localStorage.getItem("cart") || "[]")
    const existingItemIndex = cart.findIndex((item: any) => item.id === product.id)

    if (existingItemIndex > -1) {
      // Item exists, increase quantity
      cart[existingItemIndex].quantity += 1
    } else {
      // New item, add to cart
      cart.push({
        id: product.id,
        title: product.title,
        price: Math.round(product.price),
        quantity: 1,
      })
    }

    localStorage.setItem("cart", JSON.stringify(cart))

    // Trigger storage event for cart count update
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify(cart),
      }),
    )

    alert(`${product.title} added to cart!`)
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-white p-4 rounded shadow animate-pulse">
              <div className="bg-gray-200 h-48 rounded mb-4"></div>
              <div className="bg-gray-200 h-4 rounded mb-2"></div>
              <div className="bg-gray-200 h-4 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!query) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>Search for products to see results</p>
      </div>
    )
  }

  return (
    <div className="p-2 sm:p-4">
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-4">
        {products.map((product) => {
          return (
            <div key={product.id} className="bg-white rounded shadow hover:shadow-lg transition-shadow p-2 sm:p-4">
              <div className="relative mb-2 sm:mb-4">
                <img
                  src={product.images && product.images.length > 0 ? product.images[0] : "/placeholder.svg"}
                  alt={product.title}
                  loading="lazy"
                  className="w-full h-32 sm:h-48 object-cover rounded"
                />
                <button className="absolute top-1 right-1 sm:top-2 sm:right-2 p-1 bg-white rounded-full shadow">
                  <Heart size={12} className="sm:w-4 sm:h-4 text-gray-400" />
                </button>
              </div>

              <h3 className="font-medium text-xs sm:text-sm mb-1 sm:mb-2 line-clamp-2">{product.title}</h3>

              <div className="flex items-center mb-1 sm:mb-2">
                <div className="flex items-center bg-green-500 text-white px-1 py-0.5 rounded text-xs">
                  <span className="text-xs">{product.rating}</span>
                  <Star size={8} className="sm:w-2.5 sm:h-2.5 ml-1 fill-current" />
                </div>
              </div>

              <div className="mb-1 sm:mb-2">
                <span className="text-sm sm:text-lg font-semibold">
                  ₹{Math.round(product.price).toLocaleString()}
                </span>
                {product.retail_price > product.price && (
                  <span className="text-xs sm:text-sm text-gray-500 line-through ml-1 sm:ml-2">
                    ₹{product.retail_price.toLocaleString()}
                  </span>
                )}
              </div>

              <div className="space-y-1 sm:space-y-2">
                <button
                  onClick={() => addToCart(product)}
                  className="w-full bg-[#ff9f00] hover:bg-[#e68900] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors"
                >
                  Add to Cart
                </button>
                <button className="w-full bg-[#fb641b] hover:bg-[#e55a1b] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors">
                  Buy Now
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {products.length === 0 && query && (
        <div className="text-center py-8 text-gray-500">
          <p className="text-sm sm:text-base">No products found for "{query}"</p>
          <p className="text-xs sm:text-sm mt-2">Try searching with different keywords</p>
        </div>
      )}
    </div>
  )
}
